from toolchemy.nlp.clean_text import (
    clean_text,
    _clean_html,
    _normalize_unicode,
    _remove_boilerplate,
    _remove_urls,
    _clean_social_media,
    _normalize_whitespace,
)


class TestCleanHtml:
    def test_strips_tags(self):
        assert _clean_html("<p>Hello</p>") == "Hello"

    def test_strips_nested_tags(self):
        assert _clean_html("<div><b>Bold</b> text</div>") == "Bold text"

    def test_decodes_html_entities(self):
        assert _clean_html("&amp; &lt; &gt; &quot;") == '& < > "'

    def test_decodes_numeric_entities(self):
        assert _clean_html("&#8220;quote&#8221;") == "\u201cquote\u201d"

    def test_handles_mixed_tags_and_entities(self):
        assert _clean_html("<a href='#'>link &amp; text</a>") == "link & text"

    def test_preserves_plain_text(self):
        assert _clean_html("no tags here") == "no tags here"


class TestNormalizeUnicode:
    def test_nfkc_normalizes_fullwidth(self):
        assert _normalize_unicode("\uff21\uff22\uff23") == "ABC"

    def test_removes_zero_width_spaces(self):
        assert _normalize_unicode("hello\u200bworld") == "helloworld"

    def test_removes_soft_hyphen(self):
        assert _normalize_unicode("hy\u00adphen") == "hyphen"

    def test_removes_bom(self):
        assert _normalize_unicode("\ufefftext") == "text"

    def test_removes_control_characters(self):
        assert _normalize_unicode("clean\x00\x01\x02text") == "cleantext"

    def test_preserves_tab_and_newline(self):
        assert _normalize_unicode("line1\nline2\ttab") == "line1\nline2\ttab"

    def test_removes_word_joiner(self):
        assert _normalize_unicode("word\u2060joiner") == "wordjoiner"


class TestRemoveBoilerplate:
    def test_removes_read_more(self):
        result = _remove_boilerplate("Article text.\nRead more\nNext paragraph.")
        assert "Read more" not in result
        assert "Article text." in result

    def test_removes_continue_reading(self):
        result = _remove_boilerplate("Text.\nContinue Reading...\nMore text.")
        assert "Continue Reading" not in result

    def test_removes_share_lines(self):
        result = _remove_boilerplate("Content.\nShare this: Facebook Twitter\nMore.")
        assert "Share this" not in result
        assert "Content." in result

    def test_removes_subscribe_newsletter(self):
        result = _remove_boilerplate("News.\nSubscribe to our newsletter today\nMore.")
        assert "Subscribe" not in result

    def test_removes_advertisement(self):
        result = _remove_boilerplate("Text.\nAdvertisement\nMore text.")
        assert "Advertisement" not in result

    def test_removes_follow_us(self):
        result = _remove_boilerplate("News.\nFollow us on Twitter\nEnd.")
        assert "Follow us" not in result

    def test_removes_photo_count(self):
        result = _remove_boilerplate("Story.\n5 photos\nCaption.")
        assert "5 photos" not in result

    def test_removes_copyright(self):
        result = _remove_boilerplate("Article.\nAll rights reserved 2024")
        assert "All rights reserved" not in result

    def test_removes_related(self):
        result = _remove_boilerplate("Story.\nRelated:\nAnother article.")
        assert "Related:" not in result

    def test_preserves_normal_content(self):
        text = "The president announced new policy measures today."
        assert _remove_boilerplate(text) == text


class TestRemoveUrls:
    def test_removes_https(self):
        assert _remove_urls("Visit  for info") == "Visit  for info"

    def test_removes_http(self):
        assert _remove_urls("See http://example.com here") == "See  here"

    def test_removes_www(self):
        assert _remove_urls("Check www.example.com now") == "Check  now"

    def test_removes_complex_url(self):
        result = _remove_urls("Link: https://example.com/path?q=1&r=2#frag end")
        assert "https://" not in result
        assert "end" in result

    def test_removes_ftp(self):
        assert "ftp://" not in _remove_urls("Get ftp://files.example.com/doc.pdf here")

    def test_preserves_text_without_urls(self):
        text = "No links in this sentence."
        assert _remove_urls(text) == text


class TestCleanSocialMedia:
    def test_converts_hashtag_to_word(self):
        assert _clean_social_media("#Breaking") == "Breaking"

    def test_converts_multiple_hashtags(self):
        result = _clean_social_media("#News #Politics #World")
        assert result == "News Politics World"

    def test_removes_mentions(self):
        assert _clean_social_media("@reuters reports") == " reports"

    def test_removes_mentions_with_dots(self):
        assert _clean_social_media("via @bbc.news today") == "via  today"

    def test_handles_mixed_artifacts(self):
        result = _clean_social_media("@journalist: #BreakingNews in #Europe")
        assert result == ": BreakingNews in Europe"

    def test_preserves_email_like_text(self):
        result = _clean_social_media("contact@example")
        assert result == "contact"

    def test_preserves_non_social_text(self):
        text = "The 2024 election results are in."
        assert _clean_social_media(text) == text


class TestNormalizeWhitespace:
    def test_collapses_multiple_spaces(self):
        assert _normalize_whitespace("hello    world") == "hello world"

    def test_collapses_triple_newlines(self):
        assert _normalize_whitespace("a\n\n\nb") == "a\n\nb"

    def test_preserves_double_newlines(self):
        assert _normalize_whitespace("para1\n\npara2") == "para1\n\npara2"

    def test_strips_leading_trailing(self):
        assert _normalize_whitespace("  text  ") == "text"

    def test_normalizes_tabs_to_spaces(self):
        assert _normalize_whitespace("col1\tcol2") == "col1 col2"

    def test_handles_mixed_whitespace(self):
        result = _normalize_whitespace("  hello  \n\n\n\n  world  ")
        assert result == "hello\n\nworld"


class TestCleanTextPipeline:
    def test_full_pipeline(self):
        dirty = (
            "<p>Breaking news &amp; updates!</p>\n"
            "Read more at https://example.com/story\n"
            "#BreakingNews @reporter\n"
            "\u200b\n\n\n\n"
            "The actual story content here.\n"
            "Read more..."
        )
        result = clean_text(dirty)
        assert "<p>" not in result
        assert "&amp;" not in result
        assert "https://" not in result
        assert "#BreakingNews" not in result
        assert "@reporter" not in result
        assert "\u200b" not in result
        assert "The actual story content here." in result

    def test_preserves_case(self):
        assert "NATO" in clean_text("NATO announced today")

    def test_preserves_numbers(self):
        assert "42" in clean_text("The answer is 42.")

    def test_preserves_punctuation(self):
        result = clean_text("Hello, world! How are you?")
        assert "," in result
        assert "!" in result
        assert "?" in result

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_whitespace_only(self):
        assert clean_text("   \n\n\t  ") == ""

    def test_already_clean(self):
        text = "This is a perfectly clean sentence."
        assert clean_text(text) == text

    def test_realistic_rss_article(self):
        article = (
            "<div class='content'>The European Central Bank raised interest rates "
            "by 0.25% on Thursday, citing persistent inflation concerns.</div>\n"
            "Source: https://ecb.europa.eu/press/2024\n"
            "Share this: Twitter Facebook\n"
            "Related:\n"
            "All rights reserved © 2024 Reuters"
        )
        result = clean_text(article)
        assert "European Central Bank" in result
        assert "0.25%" in result
        assert "inflation" in result
        assert "https://" not in result
        assert "Share this" not in result
        assert "All rights reserved" not in result

    def test_realistic_bluesky_post(self):
        post_text = (
            "@politico Breaking: Senate passes infrastructure bill "
            "#Infrastructure #USPolitics\n"
            "Read more at https://politico.com/story/123"
        )
        result = clean_text(post_text)
        assert "Senate passes infrastructure bill" in result
        assert "Infrastructure" in result
        assert "USPolitics" in result
        assert "@politico" not in result
        assert "#Infrastructure" not in result
        assert "https://" not in result
