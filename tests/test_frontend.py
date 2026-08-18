from pathlib import Path


INDEX_HTML = Path(__file__).parents[1] / "static" / "index.html"


def test_completed_analysis_keeps_total_time_visible():
    html = INDEX_HTML.read_text(encoding="utf-8")
    success_handler = html.split("payload.status === 'succeeded'", 1)[1].split(
        "await new Promise(resolve => setTimeout(resolve, 1500));", 1
    )[0]

    assert "stopAnalysisTimer();" in success_handler
    assert "showProgress(100" in success_handler
    assert "progress.hidden = true;" not in success_handler
    assert "Total analysis time<strong id=\"analysis-time-metric\"" in html
    assert "formatCompletedElapsed(analysisElapsedMs)" in html


def test_subsecond_cached_analysis_has_a_readable_duration():
    html = INDEX_HTML.read_text(encoding="utf-8")

    assert "milliseconds < 1000 ? '<1 s'" in html
