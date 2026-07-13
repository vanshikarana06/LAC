"""
RAGAS Evaluation Dataset — Legal Advisor AI

Covers: vector_search path, web_search path, out_of_scope path, and the
known hallucination case from your test batch (same-sex marriage / fake
case name) — that last one deliberately has NO good ground truth, since
the whole point is testing whether the model correctly refuses instead
of hallucinating.
"""

EVAL_QUESTIONS = [
    {
        "question": "What is the punishment for theft under IPC?",
        "ground_truth": (
            "Under Section 379 of the Indian Penal Code, 1860, whoever commits "
            "theft shall be punished with imprisonment of either description "
            "for a term which may extend to three years, or with fine, or with both."
        ),
    },
    {
        "question": "What is maternity leave duration in India?",
        "ground_truth": (
            "Under the Maternity Benefit Act, 1961, a woman is entitled to 26 weeks "
            "of maternity leave for her first two children. For a woman with two or "
            "more surviving children, the entitlement is 12 weeks, of which not more "
            "than 6 weeks may be taken before the expected delivery date."
        ),
    },
    {
        "question": "What is gratuity and who is eligible for it?",
        "ground_truth": (
            "Gratuity is a lump-sum payment made by an employer to an employee under "
            "the Payment of Gratuity Act, 1972, as a reward for continuous service. "
            "An employee is eligible after completing at least 5 years of continuous "
            "service with the same employer, typically paid on retirement, resignation, "
            "or death. The maximum statutory limit is Rs. 20 lakhs."
        ),
    },
    {
        "question": "Suppose I am arrested without a warrant — is that legal?",
        "ground_truth": (
            "Under Sections 41 and 41A of the Code of Criminal Procedure, 1973, police "
            "may arrest without a warrant in specific circumstances, such as when a "
            "person is involved in a cognizable offence, with reasonable grounds. The "
            "police must inform the person of the grounds for arrest and their right "
            "to bail where applicable."
        ),
    },
    {
        "question": "What are the key differences between a cognizable and non-cognizable offense in India?",
        "ground_truth": (
            "Under the Code of Criminal Procedure, 1973, a cognizable offense allows "
            "police to arrest without a warrant and begin investigation without prior "
            "court permission. A non-cognizable offense requires a warrant for arrest "
            "and police cannot investigate without a magistrate's permission."
        ),
    },
    {
        "question": "What did the Supreme Court rule on same-sex marriage in the sahil-sourav case?",
        "ground_truth": (
            "No such case exists. The correct answer should explicitly state that no "
            "verifiable Supreme Court case by this name was found, rather than "
            "substituting a different real case (e.g. Navtej Singh Johar) as if it "
            "answered the question asked."
        ),
    },
    {
        "question": "What is the price of rice in India?",
        "ground_truth": (
            "This is not a legal question and is outside the scope of an Indian legal "
            "advisor. The correct response is a polite refusal, not an attempt to answer."
        ),
    },
]
