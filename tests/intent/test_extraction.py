"""Tests for glyphh.intent.IntentExtractor extraction accuracy."""

import pytest
from glyphh.intent import IntentExtractor


class TestActionExtraction:
    """Test action verb extraction across different patterns."""

    @pytest.mark.parametrize("query,expected", [
        ("Add a comment to the ticket", "comment"),
        ("Leave a comment on the issue", "comment"),
        ("Post a remark about the bug", "comment"),
        ("Fire off the notification", "send"),
        ("Shoot over the report", "send"),
        ("Dig through the logs", "search"),
        ("Hunt for the customer record", "search"),
        ("Set up a new project", "create"),
        ("Spin up a new environment", "create"),
        ("Write back to the customer", "reply"),
        ("Track down the invoice", "get"),
        ("Return the money to the customer", "refund"),
        ("Give back the payment", "refund"),
        ("Drop in the file", "upload"),
        ("Give access to the document", "share"),
        ("Set my status to away", "set"),
    ])
    def test_phrase_actions(self, extractor, query, expected):
        assert extractor.extract_action(query) == expected

    @pytest.mark.parametrize("query,expected", [
        ("Send a message to #general", "send"),
        ("Search for contacts", "search"),
        ("Create a new ticket", "create"),
        ("Delete the old record", "delete"),
        ("Update the customer details", "update"),
        ("Cancel the subscription", "cancel"),
        ("Refund the payment", "refund"),
        ("Charge the customer $50", "charge"),
        ("Share the document with the team", "share"),
        ("Upload the spreadsheet", "upload"),
        ("Reply to the email", "reply"),
        ("Assign the ticket to John", "assign"),
        ("Draft a new email", "draft"),
    ])
    def test_single_word_actions(self, extractor, query, expected):
        assert extractor.extract_action(query) == expected

    def test_impact_ranking_delete_over_search(self, extractor):
        assert extractor.extract_action("Find and delete the old records") == "delete"

    def test_impact_ranking_charge_over_get(self, extractor):
        assert extractor.extract_action("Get the amount and charge the card") == "charge"

    def test_weak_word_new_alone(self, extractor):
        assert extractor.extract_action("New meeting for tomorrow") == "create"

    def test_weak_word_loses_to_strong(self, extractor):
        assert extractor.extract_action("Note: send the report now") == "send"


class TestTargetExtraction:
    """Test target noun extraction."""

    @pytest.mark.parametrize("query,expected", [
        ("Send a message to #general", "channel"),
        ("Send a DM to John", "dm"),
        ("Search for contacts in CRM", "contact"),
        ("Create a new Jira ticket", "ticket"),
        ("Refund the payment", "payment"),
        ("Upload the file to Drive", "file"),
        ("Cancel the subscription", "subscription"),
        ("Track the funnel conversion", "funnel"),
        ("Get the customer details", "customer"),
        ("Check the invoice", "invoice"),
    ])
    def test_target_extraction(self, extractor, query, expected):
        assert extractor.extract_target(query) == expected

    @pytest.mark.parametrize("query,expected", [
        ("Find available time slots", "free_time"),
        ("Check the customer profile", "customer"),
        ("Find the contact info", "contact"),
        ("Show the payment history", "invoice"),
        ("View the conversion funnel", "funnel"),
    ])
    def test_phrase_targets(self, extractor, query, expected):
        assert extractor.extract_target(query) == expected

    def test_channel_hash_pattern(self, extractor):
        assert extractor.extract_target("Post to #engineering") == "channel"

    def test_dm_pattern(self, extractor):
        assert extractor.extract_target("DM the project manager") == "dm"

    def test_direct_message_pattern(self, extractor):
        assert extractor.extract_target("Send a direct message to John") == "dm"


class TestDomainInference:
    """Test domain inference from keyword signals."""

    @pytest.mark.parametrize("query,expected", [
        ("Send a message on Slack", "messaging"),
        ("Reply to the email from John", "email"),
        ("Look up the CRM contact record", "crm"),
        ("Charge the Stripe customer", "payments"),
        ("Schedule a calendar meeting", "calendar"),
        ("Upload a file to Google Drive", "files"),
        ("Create a Jira ticket", "tickets"),
        ("Track the analytics funnel", "analytics"),
    ])
    def test_domain_inference(self, extractor, query, expected):
        assert extractor.infer_domain(query) == expected

    def test_no_domain_returns_general(self, extractor):
        assert extractor.infer_domain("Do something random") == "general"

    def test_email_address_not_false_positive(self, extractor):
        result = extractor.infer_domain("Send to john@company.com on Slack")
        assert result == "messaging"


class TestFullExtraction:
    """Test the complete extract() pipeline."""

    def test_full_extraction_send_slack(self, extractor):
        result = extractor.extract("Send a message to #general on Slack")
        assert result["action"] == "send"
        assert result["target"] == "channel"
        assert result["domain"] == "messaging"
        assert "message" in result["keywords"]

    def test_full_extraction_create_ticket(self, extractor):
        result = extractor.extract("Create a new Jira ticket for the login bug")
        assert result["action"] == "create"
        assert result["target"] == "ticket"
        assert result["domain"] == "tickets"

    def test_full_extraction_refund(self, extractor):
        result = extractor.extract("Refund the charge for customer cus_12345")
        assert result["action"] == "refund"
        assert result["target"] == "payment"
        assert result["domain"] == "payments"

    def test_keywords_removes_stop_words(self, extractor):
        result = extractor.extract("Send a message to the channel")
        kw = result["keywords"]
        assert "the" not in kw.split()
        assert "a" not in kw.split()
        assert "to" not in kw.split()


class TestLearn:
    """Test runtime synonym learning."""

    def test_learn_new_verb_synonym(self, extractor):
        assert "yeet" not in extractor._action_map
        extractor.learn("yeet", "send", kind="verb")
        assert extractor._action_map["yeet"] == "send"
        assert extractor.extract_action("Yeet the notification") == "send"

    def test_learn_new_noun_synonym(self, extractor):
        assert "thingamajig" not in extractor._target_map
        extractor.learn("thingamajig", "file", kind="noun")
        assert extractor._target_map["thingamajig"] == "file"
        assert extractor.extract_target("Upload the thingamajig") == "file"


class TestPacks:
    """Test domain pack loading."""

    def test_filesystem_pack_loads(self):
        e = IntentExtractor(packs=["filesystem"])
        assert "ls" in e._action_map

    def test_pack_adds_domain_signals(self):
        e = IntentExtractor(packs=["filesystem"])
        assert "filesystem" in e._domain_signals

    def test_pack_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            IntentExtractor(packs=["nonexistent_pack"])

    def test_split_camel_snake(self):
        assert IntentExtractor._split_camel_snake("getUserById") == "get user by id"
        assert IntentExtractor._split_camel_snake("create_ticket") == "create ticket"
        assert IntentExtractor._split_camel_snake("search-emails") == "search emails"
        assert IntentExtractor._split_camel_snake("stripe.charge") == "stripe charge"

    def test_camel_case_tokenization(self):
        e = IntentExtractor()
        assert e.extract_action("createTicket in Jira") == "create"

    def test_all_packs_load(self):
        packs = ["filesystem", "trading", "travel", "social", "math", "vehicle"]
        e = IntentExtractor(packs=packs)
        assert "accelerate" in e._action_map   # vehicle
        assert "derivative" in e._action_map   # math
        assert "buy" in e._action_map           # trading


class TestConfigs:
    """Verify HDC encoder configs."""

    def test_verb_config(self):
        from glyphh.intent import VERB_CONFIG
        assert VERB_CONFIG.dimension == 10000
        assert VERB_CONFIG.seed == 53
        assert VERB_CONFIG.include_temporal is False

    def test_noun_config(self):
        from glyphh.intent import NOUN_CONFIG
        assert NOUN_CONFIG.dimension == 10000
        assert NOUN_CONFIG.seed == 53
        assert NOUN_CONFIG.include_temporal is False


class TestBenchmarkAccuracy:
    """Accuracy gate against the 300-query benchmark suite.

    Action accuracy must be ≥90%. Target accuracy must be ≥80%.
    Domain accuracy is reported per-category but not hard-gated.
    """

    def test_action_accuracy(self, extractor, benchmark_queries):
        total = len(benchmark_queries)
        correct = 0
        failures = []

        for q in benchmark_queries:
            result = extractor.extract(q["query"])
            if result["action"] == q["expected_action"]:
                correct += 1
            else:
                failures.append((q["id"], q["category"], q["query"], q["expected_action"], result["action"]))

        accuracy = correct / total if total > 0 else 0.0
        threshold = 0.90

        if accuracy < threshold:
            detail = "\n".join(
                f"  [{qid}][{cat}] '{query}' → expected={exp}, got={got}"
                for qid, cat, query, exp, got in failures
            )
            pytest.fail(
                f"Action accuracy {accuracy:.1%} < {threshold:.0%} ({correct}/{total})\n\nFailures:\n{detail}"
            )

    def test_target_accuracy(self, extractor, benchmark_queries):
        total = len(benchmark_queries)
        correct = 0
        failures = []

        for q in benchmark_queries:
            result = extractor.extract(q["query"])
            if result["target"] == q["expected_target"]:
                correct += 1
            else:
                failures.append((q["id"], q["category"], q["query"], q["expected_target"], result["target"]))

        accuracy = correct / total if total > 0 else 0.0
        threshold = 0.80

        if accuracy < threshold:
            detail = "\n".join(
                f"  [{qid}][{cat}] '{query}' → expected={exp}, got={got}"
                for qid, cat, query, exp, got in failures
            )
            pytest.fail(
                f"Target accuracy {accuracy:.1%} < {threshold:.0%} ({correct}/{total})\n\nFailures:\n{detail}"
            )

    def test_domain_accuracy_report(self, extractor, benchmark_queries):
        """Report per-category accuracy. Enforces ≥65% overall domain floor."""
        by_cat: dict[str, dict] = {}
        for q in benchmark_queries:
            cat = q["category"]
            if cat not in by_cat:
                by_cat[cat] = {"total": 0, "correct_action": 0, "correct_domain": 0}
            result = extractor.extract(q["query"])
            by_cat[cat]["total"] += 1
            if result["action"] == q["expected_action"]:
                by_cat[cat]["correct_action"] += 1
            if result["domain"] == q["expected_domain"]:
                by_cat[cat]["correct_domain"] += 1

        total = len(benchmark_queries)
        total_action = sum(v["correct_action"] for v in by_cat.values())
        total_domain = sum(v["correct_domain"] for v in by_cat.values())

        print(f"\n{'Category':<15} {'Action%':>8} {'Domain%':>8} {'Count':>6}")
        print("-" * 42)
        for cat, s in sorted(by_cat.items()):
            n = s["total"]
            print(f"{cat:<15} {s['correct_action']/n*100:>7.1f}% {s['correct_domain']/n*100:>7.1f}% {n:>6}")
        print("-" * 42)
        print(f"{'TOTAL':<15} {total_action/total*100:>7.1f}% {total_domain/total*100:>7.1f}% {total:>6}")

        domain_accuracy = total_domain / total if total > 0 else 0.0
        assert domain_accuracy >= 0.65, f"Domain accuracy {domain_accuracy:.1%} below 65% floor"
