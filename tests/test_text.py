from ai_isms.text import parse, stem


def test_sentence_split_handles_abbreviations_and_initials():
    doc = parse("Dr. Smith met J. Doe at 3.5 p.m. on Friday. They talked, e.g. about rust. Done!")
    assert [s.text for s in doc.sentences] == [
        "Dr. Smith met J. Doe at 3.5 p.m. on Friday.",
        "They talked, e.g. about rust.",
        "Done!",
    ]


def test_code_and_urls_are_masked_but_offsets_preserved():
    text = "Use `delve()` here.\n\n```\nIn today's fast-paced world\n```\n\nSee https://example.com/delve now."
    doc = parse(text)
    assert len(doc.masked) == len(text)
    assert "delve" not in doc.masked
    assert "fast-paced" not in doc.masked
    assert any(p.kind == "code" for p in doc.paragraphs)


def test_markdown_structure():
    doc = parse("# Title\n\nIntro line.\n\n- **One:** first\n- **Two:** second\n\n> quoted text")
    kinds = [p.kind for p in doc.paragraphs]
    assert kinds == ["heading", "prose", "list_item", "list_item", "quote"]
    assert doc.markdown["bold_lead_items"] == 2
    # Heading words are excluded from the prose word count.
    assert doc.word_count == 2 + 2 + 2 + 2


def test_stem():
    assert stem("solutions") == stem("solution")
    assert stem("changing") == stem("changed")
