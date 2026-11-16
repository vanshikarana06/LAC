import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from src.api.main import (
    is_legal_question,
    detect_question_type,
    _is_ambiguous,
    is_grounded_answer,
    is_critic_risky,
    expand_legal_acronyms,
    node_check_threshold,
    web_search_fallback,
    SIMILARITY_THRESHOLD,
    LegalAdvisorState,
)


class TestPureRoutingFunctions(unittest.TestCase):
    def test_is_legal_question_positive(self):
        positive_queries = [
            "What is the punishment for theft under IPC?",
            "Can I file an FIR with police for harassment?",
            "What are my fundamental rights under the Constitution?",
            "Can an employer fire a pregnant employee?",
            "How can a consumer claim a refund for a defective product?",
            "What are the alimony rights during a divorce in India?",
            "What is the process for property registration?",
        ]
        for query in positive_queries:
            with self.subTest(query=query):
                self.assertTrue(is_legal_question(query), f"Expected legal for: {query}")

    def test_is_legal_question_negative(self):
        negative_queries = [
            "What is the capital of France?",
            "How to bake a chocolate cake at home?",
            "What is the price of an iPhone in India?",
            "Who won the cricket match yesterday?",
            "Best places to visit in summer vacations",
        ]
        for query in negative_queries:
            with self.subTest(query=query):
                self.assertFalse(is_legal_question(query), f"Expected non-legal for: {query}")

    def test_detect_question_type_scenario(self):
        scenarios = [
            "What if my employer refuses to pay gratuity after 5 years?",
            "Suppose I am arrested without a warrant — is that legal?",
            "In a situation where a company terminates an employee without notice",
            "What happens if my landlord refuses to return the security deposit?",
        ]
        for query in scenarios:
            with self.subTest(query=query):
                self.assertEqual(detect_question_type(query), "scenario")

    def test_detect_question_type_multihop(self):
        # Touches both labour ("employee", "salary") and criminal ("fir", "police") without scenario keywords
        query = "Can an employee file an FIR against an employer for unpaid salary?"
        self.assertEqual(detect_question_type(query), "multi_hop")

    def test_detect_question_type_compound(self):
        # Long question (>80 chars) within a single domain with compound signal ("and how")
        query = (
            "What is the statutory process for purchasing and registering agricultural property "
            "and how are the stamp duty rates calculated?"
        )
        self.assertGreater(len(query), 80)
        self.assertEqual(detect_question_type(query), "compound")

    def test_detect_question_type_simple(self):
        simple_queries = [
            "What is the punishment for theft under IPC?",
            "What is gratuity and who is eligible for it?",
            "Explain Section 302 of the Indian Penal Code",
        ]
        for query in simple_queries:
            with self.subTest(query=query):
                self.assertEqual(detect_question_type(query), "simple")

    def test_is_ambiguous_within_score_window(self):
        long_query = "This is a detailed legal question containing many words to exceed four words threshold"
        # Inside [SIMILARITY_THRESHOLD - 0.05, SIMILARITY_THRESHOLD + 0.10]
        self.assertTrue(_is_ambiguous(SIMILARITY_THRESHOLD, long_query))
        self.assertTrue(_is_ambiguous(SIMILARITY_THRESHOLD - 0.02, long_query))
        self.assertTrue(_is_ambiguous(SIMILARITY_THRESHOLD + 0.08, long_query))

    def test_is_ambiguous_outside_score_window(self):
        long_query = "This is a detailed legal question containing many words to exceed four words threshold"
        # High score well above window
        self.assertFalse(_is_ambiguous(0.70, long_query))
        # Low score well below window
        self.assertFalse(_is_ambiguous(0.10, long_query))

    def test_is_ambiguous_short_query_always_true(self):
        # Queries with <= 4 words trigger planning regardless of score
        short_query = "what is law"
        self.assertTrue(_is_ambiguous(0.85, short_query))
        self.assertTrue(_is_ambiguous(0.15, short_query))

    def test_is_ambiguous_what_is_bnss(self):
        # Specifically requested test: "what is bnss?" has 3 words <= 4
        self.assertTrue(_is_ambiguous(0.50, "what is bnss?"))
        self.assertTrue(_is_ambiguous(0.10, "what is bnss?"))

    def test_expand_legal_acronyms_posh(self):
        query = "What is the committee under the POSH Act?"
        expanded = expand_legal_acronyms(query)
        self.assertIn("Sexual Harassment of Women at Workplace", expanded)
        self.assertIn("POSH", expanded)

    def test_expand_legal_acronyms_others(self):
        self.assertIn("Right to Information", expand_legal_acronyms("file an RTI"))
        self.assertIn("Code of Criminal Procedure", expand_legal_acronyms("arrest under CrPC"))
        self.assertIn("Indian Penal Code", expand_legal_acronyms("theft under IPC"))


class TestNodeCheckThreshold(unittest.TestCase):
    def setUp(self):
        self.sample_docs = [
            Document(page_content="Section 379 IPC: Punishment for theft is up to three years.", metadata={"source": "ipc.pdf", "page": 174})
        ]

    def test_high_score_unambiguous_routes_to_vector_search(self):
        state = {
            "similarity_score": 0.65,
            "standalone_question": "What is the penalty for committing theft under the Indian Penal Code?",
            "original_question": "What is the penalty for committing theft under the Indian Penal Code?",
            "retrieved_docs": self.sample_docs,
        }
        output = node_check_threshold(state)
        self.assertEqual(output["query_source"], "vector_search")
        self.assertIn("Section 379 IPC", output["context"])
        self.assertEqual(output["retrieved_docs"], self.sample_docs)

    def test_ambiguous_score_routes_to_needs_planning(self):
        # Score in ambiguous band (e.g. 0.33)
        state = {
            "similarity_score": SIMILARITY_THRESHOLD + 0.01,
            "standalone_question": "What are the rules regarding tenant eviction in Indian residential properties?",
            "original_question": "What are the rules regarding tenant eviction in Indian residential properties?",
            "retrieved_docs": self.sample_docs,
        }
        output = node_check_threshold(state)
        self.assertEqual(output["query_source"], "needs_planning")

    def test_low_score_non_legal_routes_to_needs_planning(self):
        # Non-legal question with low score -> sends to needs_planning so LLM planner decides refusal/clarification
        state = {
            "similarity_score": 0.10,
            "standalone_question": "What is the best recipe for baking chocolate cupcakes?",
            "original_question": "What is the best recipe for baking chocolate cupcakes?",
            "retrieved_docs": [],
        }
        output = node_check_threshold(state)
        self.assertEqual(output["query_source"], "needs_planning")

    @patch("src.api.main.web_search_fallback")
    def test_low_score_legal_routes_to_web_search(self, mock_web_search):
        mock_web_search.return_value = "Mocked web search results from Indian Kanoon"
        state = {
            "similarity_score": 0.12,
            "standalone_question": "What is the latest Supreme Court guideline on crypto asset regulation in India?",
            "original_question": "What is the latest Supreme Court guideline on crypto asset regulation in India?",
            "retrieved_docs": [],
        }
        output = node_check_threshold(state)
        self.assertEqual(output["query_source"], "web_search")
        self.assertEqual(output["context"], "Mocked web search results from Indian Kanoon")
        self.assertEqual(output["retrieved_docs"], [])
        mock_web_search.assert_called_once_with(state["standalone_question"])


class TestIsGroundedAnswer(unittest.TestCase):
    def test_refusal_phrases_return_false(self):
        refusal_answers = [
            "I couldn't find any relevant information in the provided context.",
            "The context does not contain any details on Section 420.",
            "There is no verifiable answer in the retrieved legal text.",
            "The case you mentioned is not mentioned in the context.",
            "I could not find provisions concerning this issue.",
            "",
            None,
        ]
        for ans in refusal_answers:
            with self.subTest(ans=ans):
                self.assertFalse(is_grounded_answer(ans), f"Expected False for refusal: {ans!r}")

    def test_grounded_answers_return_true(self):
        grounded_answers = [
            "Under Section 379 of the Indian Penal Code, theft is punishable by up to three years imprisonment.",
            "According to the Maternity Benefit Act, 1961, women are entitled to 26 weeks of paid leave.",
            "Under the Payment of Gratuity Act, an employee must complete 5 years of continuous service.",
        ]
        for ans in grounded_answers:
            with self.subTest(ans=ans):
                self.assertTrue(is_grounded_answer(ans), f"Expected True for grounded answer: {ans!r}")


class TestCriticRiskGate(unittest.TestCase):
    def test_web_search_always_risky(self):
        # Web search answers are always evaluated by critic due to measured lower faithfulness
        self.assertTrue(is_critic_risky("web_search", 0.95))
        self.assertTrue(is_critic_risky("web_search", 0.20))

    def test_low_score_vector_search_is_risky(self):
        # Low score below threshold + 0.10 is risky
        low_score = SIMILARITY_THRESHOLD + 0.05
        self.assertTrue(is_critic_risky("vector_search", low_score))

    def test_high_score_vector_search_is_not_risky(self):
        # High confidence score (>= threshold + 0.10) skips critic
        high_score = SIMILARITY_THRESHOLD + 0.25
        self.assertFalse(is_critic_risky("vector_search", high_score))

    def test_other_sources_not_risky(self):
        self.assertFalse(is_critic_risky("out_of_scope", 0.10))
        self.assertFalse(is_critic_risky("clarification", 0.10))


class TestWebSearchFallback(unittest.TestCase):
    @patch("src.api.main.TavilySearch")
    def test_tavily_raises_exception(self, MockTavilySearch):
        mock_instance = MagicMock()
        mock_instance.invoke.side_effect = Exception("Connection timed out (HTTP 504)")
        MockTavilySearch.return_value = mock_instance

        result = web_search_fallback("What is the law on noise pollution?")
        self.assertIn("Web search is currently unavailable", result)

    @patch("src.api.main.TavilySearch")
    def test_tavily_returns_plain_str(self, MockTavilySearch):
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = "No results found on legal domains"
        MockTavilySearch.return_value = mock_instance

        result = web_search_fallback("What is the law on noise pollution?")
        self.assertIn("No relevant results found from Indian legal sources", result)
        self.assertIn("No results found on legal domains", result)

    @patch("src.api.main.TavilySearch")
    def test_tavily_returns_valid_dict(self, MockTavilySearch):
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = {
            "results": [
                {"content": "Noise pollution rules 2000 regulate ambient noise.", "url": "https://indiankanoon.org/doc/123"}
            ]
        }
        MockTavilySearch.return_value = mock_instance

        result = web_search_fallback("What is the law on noise pollution?")
        self.assertIn("Web Search Results (Indian Legal Sources):", result)
        self.assertIn("Noise pollution rules 2000", result)
        self.assertIn("https://indiankanoon.org/doc/123", result)


if __name__ == "__main__":
    unittest.main()
