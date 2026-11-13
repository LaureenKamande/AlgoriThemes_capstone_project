"""Compile completed Survey Insight Studio findings into a preview and Word report.

No Gemini calls, retraining, or changes to topic/theme assignments.
"""
from __future__ import annotations

import io
from collections import OrderedDict

import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor


def compile_report(results, analysis_type, survey_question="", recommendations=None,
                   recommendation_evidence=None, max_quotes_per_theme=2):
    explorer = results["response_explorer"].copy()
    denominator = len(explorer)
    if not denominator:
        raise ValueError("The analysis has no responses to report.")
    mapping = results.get("theme_mapping")
    if not isinstance(mapping, pd.DataFrame):
        mapping = pd.DataFrame()
    response_column = results.get("response_column")
    if not response_column or response_column not in explorer.columns:
        candidates = [c for c in explorer.columns if c not in {
            "clean_text", "bert_text", "topic", "topic_id", "topic_name", "theme_name",
            "theme_id", "sentiment", "sentiment_topic_id", "word_count",
            "eligible_for_topic_model", "_studio_response_id"}]
        response_column = candidates[0] if candidates else None
    topic_id_col = "sentiment_topic_id" if analysis_type == "sentiment" and "sentiment_topic_id" in explorer else "topic"
    required = {"theme_name", "topic_name", topic_id_col}
    missing = required - set(explorer.columns)
    if missing:
        raise ValueError(f"Missing completed analysis columns: {', '.join(sorted(missing))}")

    themes_generated = results.get("themes_generated", True)
    valid = explorer.loc[
        explorer["topic_name"].notna()
        & explorer["topic_name"].astype(str).ne("Unassigned")
    ].copy()
    if themes_generated:
        valid = valid.loc[valid["theme_name"].notna() &
                          valid["theme_name"].astype(str).ne("Unassigned")]
    group_keys = ["theme_name"] if themes_generated else [topic_id_col, "topic_name"]
    # Topic-first: themes are shared across positive, neutral and negative responses.

    themes = []
    if not valid.empty:
        groups = sorted(valid.groupby(group_keys, dropna=False), key=lambda pair: -len(pair[1]))
        for group_key, theme_rows in groups:
            vals = group_key if isinstance(group_key, tuple) else (group_key,)
            meta = dict(zip(group_keys, vals))
            name = str(meta["theme_name"]) if themes_generated else str(meta["topic_name"])
            sentiment = str(meta["sentiment"]) if "sentiment" in meta else None
            topic_groups = theme_rows.groupby([topic_id_col, "topic_name"], dropna=False)
            topics = []
            for (topic_id, topic_name), rows in topic_groups:
                description = ""
                if not mapping.empty and "topic_id" in mapping:
                    matched = mapping.loc[mapping["topic_id"].astype(str) == str(topic_id)]
                    if themes_generated:
                        matched = matched.loc[matched["theme_name"].astype(str) == name]
                    if sentiment and "sentiment" in matched:
                        matched = matched.loc[matched["sentiment"].astype(str) == sentiment]
                    if not matched.empty:
                        for col in ("description", "topic_description"):
                            if col in matched and pd.notna(matched.iloc[0][col]):
                                description = str(matched.iloc[0][col]).strip()
                                if description:
                                    break
                sentiment_counts = (rows["sentiment"].value_counts().to_dict()
                                    if analysis_type == "sentiment" and "sentiment" in rows
                                    else {})
                examples_by_sentiment = {}
                if sentiment_counts and response_column:
                    for label in ("Positive", "Neutral", "Negative"):
                        examples_by_sentiment[label] = (
                            rows.loc[rows["sentiment"].eq(label), response_column]
                            .dropna().astype(str).drop_duplicates().head(2).tolist()
                        )
                topics.append({"id": str(topic_id), "name": str(topic_name),
                               "count": len(rows), "percentage": 100 * len(rows) / denominator,
                               "description": description,
                               "sentiment_counts": sentiment_counts,
                               "examples_by_sentiment": examples_by_sentiment})
            topics.sort(key=lambda x: -x["count"])
            quotes = []
            if response_column:
                values = theme_rows[response_column].dropna().astype(str)
                quotes = [s.strip() for s in values if s.strip()]
                quotes = list(OrderedDict.fromkeys(quotes))[:max_quotes_per_theme]
            themes.append({"name": name, "sentiment": sentiment,
                           "count": len(theme_rows), "percentage": 100 * len(theme_rows) / denominator,
                           "topics": topics, "quotes": quotes})

    evidence_lookup = {}
    if recommendation_evidence:
        evidence_lookup = {t["evidence_id"]: t for t in recommendation_evidence.get("topics", [])}
    actions = []
    for item in recommendations or []:
        sources = [evidence_lookup[k] for k in item.get("supporting_evidence_ids", []) if k in evidence_lookup]
        actions.append({**item, "sources": sources})
    return {"question": survey_question or results.get("survey_question", ""),
            "analysis_type": analysis_type, "total": denominator,
            "assigned": len(valid), "unassigned": denominator - len(valid),
            "themes": themes, "recommendations": actions,
            "themes_generated": themes_generated,
            "group_label": "theme" if themes_generated else "topic",
            "sentiment_confidence_threshold": results.get("sentiment_confidence_threshold", 0.70)}


def report_to_docx(report):
    document = Document()
    sec = document.sections[0]
    sec.top_margin = Inches(.75)
    sec.bottom_margin = Inches(.7)
    sec.left_margin = Inches(.85)
    sec.right_margin = Inches(.85)
    styles = document.styles
    styles["Normal"].font.name = "Aptos"
    styles["Normal"].font.size = Pt(10)
    styles["Normal"].paragraph_format.space_after = Pt(7)
    for style_name in ("Title", "Heading 1", "Heading 2"):
        styles[style_name].font.name = "Aptos Display"
        styles[style_name].font.color.rgb = RGBColor(50, 27, 69)

    document.add_heading("Survey Analysis Report", 0)
    if report["question"]:
        p = document.add_paragraph()
        p.add_run("Survey question: ").bold = True
        p.add_run(str(report["question"]))
    document.add_paragraph("Analysis route: " + (
        "Sentiment and thematic analysis" if report["analysis_type"] == "sentiment"
        else "Thematic analysis"))

    document.add_heading("1. Executive Summary", level=1)
    label = report.get("group_label", "theme")
    document.add_paragraph(
        f"The analysis included {report['total']:,} responses. "
        f"{report['assigned']:,} responses were assigned to named topics, "
        f"with {report['unassigned']:,} unassigned. "
        f"The analysis presents {len(report['themes'])} {label} groups and "
        f"produced {len(report['recommendations'])} recommendations."
    )
    if report["themes"]:
        leaders = ", ".join(t["name"] for t in report["themes"][:3])
        document.add_paragraph(f"The most frequently represented {label}s were: " + leaders + ".")

    document.add_heading("2. Overview of Findings", level=1)
    if report["themes"]:
        table = document.add_table(rows=1, cols=4)
        table.style = "Light Shading Accent 1"
        for cell, header in zip(table.rows[0].cells, ("Theme" if report.get("themes_generated", True) else "Topic", "Sentiment", "Responses", "% of all")):
            cell.text = header
        for theme in report["themes"]:
            cells = table.add_row().cells
            for cell, value in zip(cells, (theme["name"], theme["sentiment"] or "—",
                                           str(theme["count"]), f"{theme['percentage']:.1f}%")):
                cell.text = value
    else:
        document.add_paragraph("No named topics were identified.")

    document.add_heading("3. Key Findings", level=1)
    for index, theme in enumerate(report["themes"], 1):
        label = f"3.{index} {theme['name']}"
        if theme["sentiment"]:
            label += f" ({theme['sentiment']})"
        document.add_heading(label, level=2)
        document.add_paragraph(f"{theme['count']:,} responses · {theme['percentage']:.1f}% of all analysed responses")
        for topic in theme["topics"]:
            if report.get("themes_generated", True):
                p = document.add_paragraph(style="Normal")
                p.add_run(f"{topic['name']} ").bold = True
                p.add_run(f"({topic['count']:,} responses; {topic['percentage']:.1f}% of all responses).")
            if topic["description"]:
                document.add_paragraph(topic["description"])
            if report["analysis_type"] == "sentiment":
                breakdown = topic.get("sentiment_counts", {})
                if breakdown:
                    document.add_paragraph("Sentiment within topic: " + "; ".join(
                        f"{label}: {breakdown.get(label, 0)} "
                        f"({100 * breakdown.get(label, 0) / max(topic['count'], 1):.1f}%)"
                        for label in ("Positive", "Neutral", "Negative")
                    ))
                for label, examples in topic.get("examples_by_sentiment", {}).items():
                    if examples:
                        document.add_paragraph(f"{label} examples:")
                        for example in examples:
                            document.add_paragraph(f'“{example}”', style="Quote")
        if theme["quotes"]:
            document.add_paragraph("Illustrative responses:").runs[0].bold = True
            for quote in theme["quotes"]:
                document.add_paragraph(f'“{quote}”', style="Quote")

    document.add_heading("4. Recommendations", level=1)
    if report["recommendations"]:
        for index, rec in enumerate(report["recommendations"], 1):
            document.add_heading(f"4.{index} {rec['title']}", level=2)
            document.add_paragraph("Priority: " + rec["priority"])
            document.add_paragraph(rec["action"])
            document.add_paragraph("Rationale: " + rec["rationale"])
            document.add_paragraph("Priority rationale: " + rec["priority_reason"])
            if rec["sources"]:
                document.add_paragraph("Supporting topics: " + "; ".join(
                    (f"{t['topic_name']} ({t['theme_name']})" if t.get("theme_name")
                     else t['topic_name']) for t in rec["sources"]))
    else:
        document.add_paragraph("Recommendations were not available for this analysis.")

    if report["analysis_type"] == "sentiment":
        document.add_paragraph(
            "Sentiment was classified using the pretrained "
            "cardiffnlp/twitter-roberta-base-sentiment-latest model. "
            "Confidence is the model's highest class probability, not a "
            "validated accuracy estimate. Responses below the review "
            f"threshold ({100 * report.get('sentiment_confidence_threshold', 0.70):g}%) "
            "retain their predicted sentiment and are flagged for human review. "
            "BERTopic was fitted once to all retained responses before "
            "sentiment classification; responses with fewer than 3 cleaned "
            "words were excluded from both analyses. Sentiment is response-level, so mixed "
            "opinions within a single response may not be captured."
        )
    document.add_paragraph(
        "Theme grouping was applied only when BERTopic identified more than 8 "
        "topics. When 8 or fewer topics were found, topic findings were "
        "reported directly without creating themes."
    )
    document.add_paragraph(
        "Interpretation note: Percentages use all analysed responses as the denominator. "
        "Topic assignments are model-derived and may include unassigned responses. "
        "The findings are descriptive and do not establish statistical significance "
        "or representativeness of a wider population."
    )
    output = io.BytesIO()
    document.save(output)
    return output.getvalue()
