"""Exercise cash-buyer pushes through the CLI, HTTP parser, and local schema."""

import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from click.testing import CliRunner

from cash_buyer_intel import cli, db


def acknowledgement(total, created, updated=0, rejected=()):
    return {
        "ok": True,
        "data": {"created": created, "updated": updated, "errors": len(rejected)},
        "meta": {"total": total, "errors": [
            {"index": index, "error": f"rejected {index}"} for index in rejected
        ]},
    }


class PushTranchiTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.ids = [f"buyer-{index}" for index in range(8)]
        self.runner = CliRunner()
        for patcher in (
            patch.object(db, "BUYERS_DB", Path(directory.name) / "buyers.db"),
            patch.object(cli, "open_buyers", lambda read_only=False: db.open_buyers(
                read_only=read_only, attach_sources=False)),
            patch.object(cli, "_load_tranchi_token", return_value="isolated-test-token"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        db.init_db()
        with db.open_buyers(attach_sources=False) as conn:
            conn.executemany(
                """INSERT INTO buyer_entities
                   (entity_id, canonical_name, entity_type, first_seen, last_seen, total_sales)
                   VALUES (?, ?, 'LLC', '2026-01-01', '2026-10-01', 5)""",
                [(entity_id, f"Buyer {index} LLC") for index, entity_id in enumerate(self.ids)],
            )
            conn.executemany(
                """INSERT INTO buyer_scores
                   (entity_id, velocity_12m, velocity_3m, recency_score, activity_tier)
                   VALUES (?, 5, 2, ?, 'hot')""",
                [(entity_id, 100 - index) for index, entity_id in enumerate(self.ids)],
            )
            conn.executemany(
                "INSERT INTO buyer_contacts (entity_id, primary_phone, primary_email) VALUES (?, ?, ?)",
                [(entity_id, f"555010000{index}", f"{entity_id}@example.test")
                 for index, entity_id in enumerate(self.ids)],
            )
            conn.commit()

    def push(self, replies, *, top=3, batch_size=3, agent=True, exit_code=0):
        batches = []

        def urlopen(request, timeout):
            self.assertEqual(request.full_url, "https://tranchi.ai/api/cash_buyers")
            self.assertEqual(request.get_method(), "POST")
            self.assertEqual(request.get_header("Authorization"), "Bearer isolated-test-token")
            self.assertEqual(request.get_header("Content-type"), "application/json")
            self.assertEqual(timeout, 30.0)
            wrapper = json.loads(request.data.decode("utf-8"))
            self.assertEqual(set(wrapper), {"buyers"})
            self.assertIsInstance(wrapper["buyers"], list)
            for record in wrapper["buyers"]:
                index = self.ids.index(record["external_id"])
                self.assertEqual(record["name"], f"Buyer {index} LLC")
                self.assertEqual(record["phone"], f"555010000{index}")
                self.assertEqual(record["email"], f"buyer-{index}@example.test")
                self.assertEqual(record["source"], "cash-buyer-scraper")
                self.assertIsNone(record["market"])
                self.assertIsNone(record["state"])
            batches.append([record["external_id"] for record in wrapper["buyers"]])
            self.assertLessEqual(len(batches), len(replies), "unexpected POST or retry")
            reply = replies[len(batches) - 1]
            if isinstance(reply, BaseException):
                raise reply
            return io.BytesIO(reply if isinstance(reply, bytes) else json.dumps(reply).encode("utf-8"))

        args = ["push-tranchi", "--top", str(top), "--batch-size", str(batch_size)]
        if agent:
            args.append("--agent")
        with patch("urllib.request.urlopen", side_effect=urlopen) as mocked:
            result = self.runner.invoke(cli.main, args)
        self.assertEqual(mocked.call_count, len(replies), result.output)
        self.assertEqual(batches, [self.ids[start:min(start + batch_size, top)]
                                   for start in range(0, top, batch_size)], result.output)
        self.assertEqual(result.exit_code, exit_code, f"{result.exception!r}\n{result.output}")
        if not agent:
            return result
        envelope = json.loads(result.output)
        self.assertEqual(set(envelope), {"ok", "data", "errors", "meta"})
        self.assertIsInstance(envelope["errors"], list)
        self.assertEqual(envelope["meta"]["batch_size"], batch_size)
        return envelope

    def assert_data(self, result, created, updated, rejected, *, candidates=3, errors=()):
        self.assertEqual(result["data"], {
            "created": created, "updated": updated, "errors": rejected,
            "candidates": candidates, "per_record_errors": list(errors),
        })

    def assert_outreach(self, entity_ids):
        with db.open_buyers(attach_sources=False) as conn:
            rows = conn.execute(
                """SELECT entity_id, channel, direction, response_status, summary
                   FROM buyer_outreach ORDER BY outreach_id"""
            ).fetchall()
        self.assertEqual([tuple(row) for row in rows], [
            (entity_id, "tranchi_push", "out", "pushed", "POST /api/cash_buyers (direct)")
            for entity_id in entity_ids
        ])

    def remaining_buyers(self):
        result = self.runner.invoke(cli.main, ["buyers", "--no-recent-outreach", "30d", "--agent"])
        self.assertEqual(result.exit_code, 0, result.output)
        envelope = json.loads(result.output)
        self.assertIs(envelope["ok"], True)
        self.assertEqual(envelope["errors"], [])
        return [row["entity_id"] for row in envelope["data"]]

    def assert_unconfirmed(self, reply, *, top=3):
        with db.open_buyers(attach_sources=False) as conn:
            conn.execute("DELETE FROM buyer_outreach")
            conn.commit()
        result = self.push([reply], top=top)
        self.assertIs(result["ok"], False)
        self.assert_data(result, 0, 0, 0, candidates=top)
        self.assertEqual(len(result["errors"]), 1)
        self.assertIn("unconfirmed", result["errors"][0])
        self.assert_outreach([])

    def test_create_then_update_only_rerun_are_both_confirmed(self):
        created = self.push([acknowledgement(3, 3)])
        self.assertIs(created["ok"], True)
        self.assertEqual(created["errors"], [])
        self.assert_data(created, 3, 0, 0)
        self.assert_outreach(self.ids[:3])
        self.assertEqual(self.remaining_buyers(), self.ids[3:])

        updated = self.push([acknowledgement(3, 0, 3)])
        self.assertIs(updated["ok"], True)
        self.assertEqual(updated["errors"], [])
        self.assert_data(updated, 0, 3, 0)
        self.assert_outreach(self.ids[:3] * 2)
        self.assertEqual(self.remaining_buyers(), self.ids[3:])

    def test_partial_rejection_boundary_indices_preserve_only_accepted_outreach(self):
        result = self.push([acknowledgement(3, 0, 1, (0, 2))])
        self.assertIs(result["ok"], True)
        self.assertEqual(result["errors"], [])
        self.assert_data(result, 0, 1, 2, errors=[
            {"external_id": self.ids[index], "error": f"rejected {index}"} for index in (0, 2)
        ])
        self.assert_outreach([self.ids[1]])
        self.assertEqual(self.remaining_buyers(), [self.ids[0]] + self.ids[2:])

    def test_all_rejected_server_success_is_not_local_success(self):
        result = self.push([acknowledgement(3, 0, 0, (0, 1, 2))])
        self.assertIs(result["ok"], False)
        self.assertEqual(result["errors"], [])
        self.assert_data(result, 0, 0, 3, errors=[
            {"external_id": self.ids[index], "error": f"rejected {index}"} for index in range(3)
        ])
        self.assert_outreach([])
        self.assertEqual(self.remaining_buyers(), self.ids)

    def test_absent_malformed_and_noninteger_acknowledgement_fields_are_unconfirmed(self):
        valid = acknowledgement(3, 1, 1, (2,))
        missing = object()
        cases = [("root", value) for value in (None, [], True, 1, "not an envelope")]
        fields = [
            (("ok",), (None, False, 0, 1, "true", [])),
            (("data",), (None, [], False, 1, "not data")),
            (("meta",), (None, [], False, 1, "not meta")),
            (("data", "created"), (-1, True, False, "1", 1.0, None, [], {})),
            (("data", "updated"), (-1, True, False, "1", 1.0, None, [], {})),
            (("data", "errors"), (-1, True, False, "1", 1.0, None, [], {})),
            (("meta", "total"), (-1, 0, 2, 4, True, False, "3", 3.0, None, [])),
            (("meta", "errors"), (None, {}, "not a list", False, [])),
        ]
        for path, values in fields:
            for value in (missing, *values):
                response = copy.deepcopy(valid)
                parent = response if len(path) == 1 else response[path[0]]
                if value is missing:
                    del parent[path[-1]]
                else:
                    parent[path[-1]] = value
                cases.append((f"{'.'.join(path)}={value!r}", response))
        cases.extend([
            ("counts below total", acknowledgement(3, 2)),
            ("counts exceed total", acknowledgement(3, 4)),
            ("malformed JSON", b'{"ok":true'),
            ("empty JSON", b""),
            ("malformed UTF8", b"\xff"),
        ])
        for label, reply in cases:
            with self.subTest(case=label):
                self.assert_unconfirmed(reply)
        self.assertEqual(self.remaining_buyers(), self.ids)

    def test_boolean_total_matching_single_record_is_still_invalid(self):
        reply = acknowledgement(1, 1)
        reply["meta"]["total"] = True
        self.assert_unconfirmed(reply, top=1)

    def test_error_list_must_be_complete_unique_and_strictly_indexed(self):
        valid = acknowledgement(3, 1, 0, (0, 2))
        first, last = valid["meta"]["errors"]
        lists = [("missing all", []), ("truncated", [first]),
                 ("extra entry", [first, last, {"index": 1, "error": "extra"}])]
        entries = [None, [], "not an entry", {}, {"index": 0}, {"error": "missing index"}]
        entries.extend({"index": index, "error": "rejected 0"}
                       for index in (-1, 3, True, False, "0", 0.0, None, 2))
        entries.extend({"index": 0, "error": error}
                       for error in (None, False, 0, [], {}, "", " \t"))
        lists.extend((f"entry={entry!r}", [entry, last]) for entry in entries)
        for label, errors in lists:
            with self.subTest(case=label):
                response = copy.deepcopy(valid)
                response["meta"]["errors"] = errors
                self.assert_unconfirmed(response)

    def test_transport_and_server_failures_do_not_confirm_possible_remote_writes(self):
        failures = [
            URLError("offline"),
            OSError("connection reset"),
            HTTPError("https://tranchi.ai/api/cash_buyers", 500, "handler failed", {},
                      io.BytesIO(b"failure after a possible remote upsert")),
        ]
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                self.assert_unconfirmed(failure)
        self.assertEqual(self.remaining_buyers(), self.ids)

    def test_mixed_batches_keep_confirmed_counts_rows_and_batch_relative_mapping(self):
        unknown = acknowledgement(3, 3)
        unknown["ok"] = False
        result = self.push([
            acknowledgement(3, 2, 1), unknown, acknowledgement(2, 0, 1, (0,)),
        ], top=8)
        self.assertIs(result["ok"], False)
        self.assertEqual(len(result["errors"]), 1)
        self.assertIn("unconfirmed", result["errors"][0])
        self.assert_data(result, 2, 2, 1, candidates=8, errors=[
            {"external_id": self.ids[6], "error": "rejected 0"},
        ])
        self.assert_outreach(self.ids[:3] + [self.ids[7]])
        self.assertEqual(self.remaining_buyers(), self.ids[3:7])

    def test_human_failure_exits_one_without_suppressing_uncertain_buyers(self):
        reply = acknowledgement(3, 3)
        reply["meta"]["total"] = 99
        result = self.push([reply], agent=False, exit_code=1)
        self.assertIn("error:", result.output)
        self.assertIn("unconfirmed", result.output)
        self.assert_outreach([])
        self.assertEqual(self.remaining_buyers(), self.ids)


if __name__ == "__main__":
    unittest.main()
