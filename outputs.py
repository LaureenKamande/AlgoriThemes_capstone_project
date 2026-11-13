"""AlgoriThemes — notebook-derived functions with Streamlit adapters.
Source: application_code_fixed(1).ipynb. Original functions retained verbatim.
"""
import io
import json
import zipfile
from pathlib import Path
from datetime import date
import pandas as pd
import plotly.express as px
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

def export_word_report(report, quote_bank, evidence, output_path):

    doc = Document()

    # Page layout
    section = doc.sections[0]
    section.top_margin = Inches(0.85)
    section.bottom_margin = Inches(0.85)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)

    # Default text
    normal = doc.styles["Normal"]
    normal.font.name = "Aptos"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(8)

    # Heading formatting
    for name, size in [
        ("Heading 1", 16),
        ("Heading 2", 13),
        ("Heading 3", 11)
    ]:
        style = doc.styles[name]
        style.font.name = "Aptos Display"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(30, 58, 95)

    # Title page
    title = doc.add_heading(
        "Qualitative Survey Analysis Report", 0
    )
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    subtitle = doc.add_paragraph(
        evidence["survey_question"]
    )
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER

    date_paragraph = doc.add_paragraph(
        date.today().strftime("%B %Y")
    )
    date_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph(
        f"Total responses analysed: "
        f"{evidence['total_responses']:,}"
    ).alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_page_break()

    # Executive summary
    doc.add_heading("Executive Summary", level=1)
    doc.add_paragraph(report["executive_summary"])

    # Key findings
    doc.add_heading("Key Findings", level=1)

    for theme in report["key_findings"]:

        doc.add_heading(
            theme["theme_name"], level=2
        )

        doc.add_paragraph(theme["theme_overview"])

        doc.add_paragraph(
            f"Responses: {theme['response_count']} "
            f"({theme['percentage']}%)"
        )

        for topic in theme["topic_findings"]:

            doc.add_heading(
                topic["topic_name"], level=3
            )

            doc.add_paragraph(topic["analysis"])

            # Exact respondent quotations
            for quote_id in topic["supporting_quote_ids"]:

                if quote_id not in quote_bank:
                    raise ValueError(
                        f"Unknown quotation: {quote_id}"
                    )

                if not quote_id.startswith(
                    f"T{topic['topic_id']}_R"
                ):
                    raise ValueError(
                        f"Quotation does not match topic: {quote_id}"
                    )

                quote = doc.add_paragraph(style="Quote")
                quote.add_run(
                    f"“{quote_bank[quote_id]}”"
                ).italic = True

                doc.add_paragraph(
                    f"Source reference: {quote_id}"
                )

        if theme.get("contrasting_perspectives"):
            doc.add_heading(
                "Variation in Responses", level=3
            )
            doc.add_paragraph(
                theme["contrasting_perspectives"]
            )

        doc.add_heading(
            "Interpretation and Implications", level=3
        )
        doc.add_paragraph(theme["interpretation"])

    # Recommendations
    doc.add_heading("Recommendations", level=1)

    for i, rec in enumerate(
        report["recommendations"], start=1
    ):
        doc.add_heading(
            f"{i}. {rec['recommendation']}",
            level=2
        )

        doc.add_paragraph(
            f"Related theme: {rec['supporting_theme']}"
        )

        doc.add_paragraph(rec["rationale"])

        doc.add_paragraph(
            f"Priority: {rec['priority']}"
        )

    # Methodological note
    doc.add_heading("Methodological Note", level=1)

    doc.add_paragraph(
        "Responses were grouped using BERTopic. "
        "Topic names and broader themes were developed "
        "through AI-assisted interpretation of sampled "
        "survey responses. Each response was assigned "
        "one primary topic. The findings and quotations "
        "should be reviewed before publication."
    )

    # Save
    doc.save(output_path)

    return output_path

def export_sentiment_word_report(
    report,
    quote_bank,
    evidence,
    output_path
):
    doc = Document()

    section = doc.sections[0]
    section.top_margin = Inches(0.85)
    section.bottom_margin = Inches(0.85)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)

    normal = doc.styles["Normal"]
    normal.font.name = "Aptos"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(8)

    for style_name, size in [
        ("Heading 1", 16),
        ("Heading 2", 13)
    ]:
        style = doc.styles[style_name]
        style.font.name = "Aptos Display"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(30, 58, 95)

    title = doc.add_heading(
        "Sentiment and Thematic Analysis Report", 0
    )
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    subtitle = doc.add_paragraph(
        evidence["survey_question"]
    )
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph(
        date.today().strftime("%B %Y")
    ).alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph(
        f"Responses analysed: "
        f"{evidence['total_responses']:,}"
    )

    # Executive summary
    doc.add_heading("Executive Summary", level=1)
    doc.add_paragraph(report["executive_summary"])

    # Overview
    doc.add_heading("Overview of Findings", level=1)

    table = doc.add_table(rows=1, cols=3)
    table.style = "Light Shading Accent 1"

    headers = ["Sentiment", "Responses", "Percentage"]

    for i, heading in enumerate(headers):
        table.rows[0].cells[i].text = heading

    for item in report["overview"]:
        cells = table.add_row().cells
        cells[0].text = str(item["sentiment"])
        cells[1].text = str(item["response_count"])
        cells[2].text = f"{item['percentage']:.1f}%"

    # Key findings
    doc.add_heading("Key Findings", level=1)

    for finding in report["key_findings"]:

        doc.add_heading(
            f"{finding['sentiment']}: "
            f"{finding['theme_name']}",
            level=2
        )

        doc.add_paragraph(finding["analysis"])

        for quote_id in finding["supporting_quote_ids"]:

            if quote_id not in quote_bank:
                raise ValueError(
                    f"Unknown quotation ID: {quote_id}"
                )

            paragraph = doc.add_paragraph(style="Quote")
            paragraph.add_run(
                f"“{quote_bank[quote_id]}”"
            ).italic = True

    # Recommendations
    doc.add_heading("Recommendations", level=1)

    for i, rec in enumerate(
        report["recommendations"], start=1
    ):

        doc.add_heading(
            f"{i}. {rec['recommendation']}",
            level=2
        )

        doc.add_paragraph(rec["rationale"])

        supporting = rec.get("supporting_themes", [])

        if supporting:
            doc.add_paragraph(
                "Related themes: " + "; ".join(supporting)
            )

    # Methodological note
    doc.add_heading("Methodological Note", level=1)

    doc.add_paragraph(
        "Sentiment labels were generated using pretrained RoBERTa; predictions below 70% model confidence were flagged for review. "
        "BERTopic identified topics across all eligible responses "
        "before RoBERTa sentiment was compared within topics. "
        "A large language model was used to name topics, "
        "group related topics into themes, and draft the "
        "report. Findings should be reviewed against the "
        "original responses before use in decision-making."
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc.save(output_path)

    return output_path


def get_theme_distribution(results):
    explorer = results["response_explorer"]
    assigned = explorer[explorer["theme_name"] != "Unassigned"]
    distribution = assigned.groupby("theme_name").size().reset_index(name="Count")
    distribution["Percentage"] = (
        100 * distribution["Count"] / max(len(explorer), 1)
    ).round(2)
    return distribution.sort_values("Count", ascending=False)


def get_sentiment_distribution(results):
    explorer = results["response_explorer"]
    dist = explorer.groupby("sentiment").size().reset_index(name="Count")
    dist["Percentage"] = (100 * dist["Count"] / max(len(explorer), 1)).round(2)
    return dist


def get_sentiment_theme_distribution(results):
    explorer = results["response_explorer"]
    assigned = explorer[explorer["theme_name"] != "Unassigned"]
    return assigned.groupby(["sentiment", "theme_name"]).size().reset_index(name="Count")


def make_theme_chart(distribution):
    return px.bar(distribution, x="Count", y="theme_name", orientation="h", title="Themes")


def make_sentiment_chart(distribution):
    return px.pie(distribution, names="sentiment", values="Count", title="Sentiment distribution")


def dataframe_to_csv(df):
    return df.to_csv(index=False).encode("utf-8-sig")


def dataframe_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")
    return output.getvalue()


def prepare_report_evidence(results, survey_question=None, analysis_type="topic"):
    """Notebook evidence structure adapted to accept a results dictionary, not globals."""
    explorer = results["response_explorer"]
    survey_question = survey_question or results.get("survey_question", "")
    if (analysis_type == "sentiment" and
            results.get("analysis_sequence") == "topic_then_sentiment"):
        # The current app uses report_builder for its Word report. For legacy
        # notebook exports, retain topic-based evidence and a sentiment overview.
        assigned = explorer.loc[explorer["theme_name"].ne("Unassigned")]
        return ({
            "survey_question": survey_question,
            "total_responses": len(explorer),
            "sentiment_distribution": get_sentiment_distribution(results).to_dict("records"),
            "topic_sentiment_distribution": results.get(
                "sentiment_by_topic", pd.DataFrame()
            ).to_dict("records"),
            "assigned_responses": len(assigned),
        }, {})
    if analysis_type == "sentiment" or "sentiment" in explorer.columns:
        evidence = {
            "survey_question": survey_question,
            "total_responses": int(len(explorer)),
            "sentiment_distribution": get_sentiment_distribution(results).to_dict("records"),
            "sentiment_groups": [],
        }
        quote_bank = {}
        sample_sets = results.get("gemini_samples", {})
        for sentiment_label in ["Positive", "Neutral", "Negative"]:
            group = explorer[explorer["sentiment"] == sentiment_label]
            group_evidence = {"sentiment": sentiment_label, "response_count": int(len(group)), "themes": []}
            assigned = group[group["theme_name"] != "Unassigned"]
            for theme_name, theme_group in assigned.groupby("theme_name"):
                theme_record = {
                    "theme_name": theme_name,
                    "response_count": int(len(theme_group)),
                    "percentage_within_sentiment": round(100 * len(theme_group) / max(len(group), 1), 2),
                    "topics": [],
                }
                for topic_id, topic_group in theme_group.groupby("sentiment_topic_id"):
                    samples = sample_sets.get(sentiment_label, pd.DataFrame())
                    sampled = samples[samples["sentiment_topic_id"] == topic_id] if not samples.empty else pd.DataFrame()
                    quotes = []
                    for _, sample in sampled.iterrows():
                        quote_id = f"{sentiment_label[0]}_T{topic_id}_R{int(sample['sentiment_response_id'])}"
                        response_text = str(sample["response"])
                        quote_bank[quote_id] = response_text
                        quotes.append({"quote_id": quote_id, "response": response_text})
                    theme_record["topics"].append({
                        "topic_id": int(topic_id),
                        "topic_name": topic_group["topic_name"].iloc[0],
                        "response_count": int(len(topic_group)),
                        "sample_responses": quotes,
                    })
                group_evidence["themes"].append(theme_record)
            evidence["sentiment_groups"].append(group_evidence)
        return evidence, quote_bank

    assigned = explorer[explorer["theme_name"] != "Unassigned"].copy()
    evidence = {
        "survey_question": survey_question,
        "total_responses": int(len(explorer)),
        "assigned_responses": int(len(assigned)),
        "themes": [],
    }
    quote_bank = {}
    theme_input = {int(t["topic_id"]): t for t in results.get("theme_input", [])}
    for theme_name, group in assigned.groupby("theme_name"):
        topic_details = []
        for topic_id, topic_group in group.groupby("topic"):
            original_topic = theme_input.get(int(topic_id), {})
            indexed = []
            for i, response in enumerate(original_topic.get("sample_responses", []), start=1):
                quote_id = f"T{topic_id}_R{i}"
                quote_bank[quote_id] = str(response)
                indexed.append({"quote_id": quote_id, "response": str(response)})
            topic_details.append({
                "topic_id": int(topic_id),
                "topic_name": original_topic.get("topic_name", topic_group["topic_name"].iloc[0]),
                "response_count": int(len(topic_group)),
                "sample_responses": indexed,
            })
        evidence["themes"].append({
            "theme_name": theme_name,
            "response_count": int(len(group)),
            "percentage": round(100 * len(group) / max(len(explorer), 1), 1),
            "topics": topic_details,
        })
    return evidence, quote_bank


def report_to_word(report, results=None, analysis_type="topic", evidence=None, quote_bank=None):
    if isinstance(report, dict) and {"themes", "total", "analysis_type"}.issubset(report):
        from report_builder import report_to_docx
        return report_to_docx(report)
    """App adapter. Use original notebook Word exporters when their expected schema is present.

    The notebook itself flags a mismatch between its main report JSON and its main
    Word exporter; this function uses a simple safe fallback for that case.
    """
    if evidence is None and results is not None:
        evidence, quote_bank = prepare_report_evidence(results, analysis_type=analysis_type)
    evidence = evidence or {"survey_question": "", "total_responses": 0}
    quote_bank = quote_bank or {}
    import tempfile
    with tempfile.TemporaryDirectory() as tempdir:
        path = Path(tempdir) / "report.docx"
        if analysis_type == "sentiment":
            try:
                export_sentiment_word_report(report, quote_bank, evidence, path)
                return path.read_bytes()
            except (KeyError, TypeError, IndexError):
                pass
        else:
            try:
                export_word_report(report, quote_bank, evidence, path)
                return path.read_bytes()
            except (KeyError, TypeError, IndexError):
                pass
        doc = Document()
        doc.add_heading("AlgoriThemes — Analysis Report", 0)
        doc.add_paragraph(evidence.get("survey_question", ""))
        def write_obj(value, level=1):
            if isinstance(value, dict):
                for key, item in value.items():
                    doc.add_heading(str(key).replace("_", " ").title(), level=min(level, 4))
                    write_obj(item, level + 1)
            elif isinstance(value, list):
                for item in value:
                    write_obj(item, level)
            else:
                doc.add_paragraph(str(value))
        write_obj(report)
        doc.save(path)
        return path.read_bytes()


def create_results_zip(results, report=None, analysis_type="topic"):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        explorer = results["response_explorer"].copy()
        internal = [c for c in (
            "positive_score", "neutral_score", "negative_score", "compound",
            "sentiment_margin", "sentiment_chunks"
        ) if c in explorer.columns]
        archive.writestr("response_explorer.csv", dataframe_to_csv(explorer.drop(columns=internal)))
        if "theme_mapping" in results and not results["theme_mapping"].empty:
            archive.writestr("theme_mapping.csv", dataframe_to_csv(results["theme_mapping"]))
        if report is not None:
            archive.writestr("analysis_report.docx", report_to_word(report, results, analysis_type))
    return buffer.getvalue()
