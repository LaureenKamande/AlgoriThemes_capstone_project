"""Standalone, offline, interactive HTML export of completed AlgoriThemes results."""
from __future__ import annotations

import html
import json
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from report_builder import compile_report

COLORS = ['#7655B8','#3B8FAD','#D08A43','#4D9B78','#C16B96','#6878C7','#9C785A','#728F45']
SENTIMENTS = ['Positive','Neutral','Negative']


def _json(value):
    return json.dumps(value, ensure_ascii=False, default=str).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')


def _records(df):
    return json.loads(df.to_json(orient='records', date_format='iso'))


def _chart(fig, identifier):
    if fig is None:
        return ''
    spec = json.loads(fig.to_json())
    return f'<div class="chart" id="{identifier}"></div><script type="application/json" id="data-{identifier}">{_json(spec)}</script>'


def _distribution(explorer, group_col, denominator, title, colors=None):
    valid = explorer.loc[explorer[group_col].notna() & explorer[group_col].astype(str).ne('Unassigned')]
    if valid.empty:
        return None
    dist = valid.groupby(group_col).size().reset_index(name='Count')
    dist['Percentage'] = dist['Count'] * 100 / max(denominator, 1)
    dist = dist.sort_values('Percentage')
    fig = px.bar(dist, x='Percentage', y=group_col, orientation='h', color=group_col,
                 color_discrete_sequence=COLORS, text='Percentage', custom_data=['Count'], title=title)
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside',
                      hovertemplate='%{y}<br>%{x:.1f}% of responses<br>n=%{customdata[0]}<extra></extra>')
    fig.update_layout(showlegend=False, height=max(420, 54*len(dist)+160), margin=dict(l=350,r=95,t=65,b=45),
                      xaxis=dict(ticksuffix='%', range=[0,max(10,float(dist['Percentage'].max())*1.25)]), yaxis_title=None)
    return fig


def _topic_chart(explorer, denominator, title):
    valid = explorer.loc[explorer['theme_name'].notna() & explorer['topic_name'].notna() &
                         explorer['theme_name'].astype(str).ne('Unassigned') & explorer['topic_name'].astype(str).ne('Unassigned')]
    if valid.empty:
        return None
    keys = ['theme_name','topic_name']
    if 'sentiment_topic_id' in valid: keys.append('sentiment_topic_id')
    elif 'topic' in valid: keys.append('topic')
    dist = valid.groupby(keys, dropna=False).size().reset_index(name='Count')
    dist['Percentage'] = dist['Count'] * 100 / max(denominator,1)
    dist = dist.sort_values('Percentage')
    fig = px.bar(dist,x='Percentage',y='topic_name',orientation='h',color='theme_name',
                 color_discrete_sequence=COLORS,text='Percentage',custom_data=['Count'],title=title)
    fig.update_traces(texttemplate='%{text:.1f}%',textposition='outside',
                      hovertemplate='%{y}<br>%{x:.1f}% of responses<br>n=%{customdata[0]}<extra></extra>')
    fig.update_layout(height=max(480,48*len(dist)+280), margin=dict(l=350,r=95,t=65,b=45),
                      xaxis=dict(ticksuffix='%',range=[0,max(5,float(dist['Percentage'].max())*1.25)]),yaxis_title=None,legend_title_text='Parent theme', legend=dict(orientation="h", y=-0.18, x=0))
    return fig


def _theme_sections(explorer, mapping, denominator, prefix, response_col, id_col):
    assigned = explorer.loc[explorer['theme_name'].notna() & explorer['topic_name'].notna() &
                            explorer['theme_name'].astype(str).ne('Unassigned') & explorer['topic_name'].astype(str).ne('Unassigned')]
    if assigned.empty:
        return '<p>No named themes were identified.</p>'
    parts = []
    topic_col = 'sentiment_topic_id' if 'sentiment_topic_id' in assigned else 'topic'
    for n,(theme_name,count) in enumerate(assigned.groupby('theme_name').size().sort_values(ascending=False).items(),1):
        group = assigned.loc[assigned['theme_name']==theme_name]
        t = group.groupby([topic_col,'topic_name'],dropna=False).size().reset_index(name='Count').sort_values('Count',ascending=False)
        t['Within Theme (%)'] = 100*t['Count']/max(count,1)
        t['All Responses (%)'] = 100*t['Count']/max(denominator,1)
        cid = f'{prefix}-theme-{n}'
        fig = px.bar(t.sort_values('Within Theme (%)'),x='Within Theme (%)',y='topic_name',orientation='h',
                     text='Within Theme (%)',custom_data=['Count','All Responses (%)'])
        fig.update_traces(marker_color=COLORS[(n-1)%len(COLORS)],texttemplate='%{text:.1f}%',textposition='outside',
                          hovertemplate='%{y}<br>%{x:.1f}% of this theme<br>n=%{customdata[0]}<br>%{customdata[1]:.1f}% of all responses<extra></extra>')
        fig.update_layout(xaxis=dict(range=[0,105],ticksuffix='%'),yaxis_title=None,showlegend=False,
                          height=max(300,len(t)*60+140),margin=dict(l=350,r=95,t=55,b=40))
        parts.append(f'<section class="theme-card"><p class="eyebrow">THEME {n}</p><h2>{html.escape(str(theme_name))}</h2>'
                     f'<div class="metrics"><span><b>{count:,}</b> responses</span><span><b>{100*count/max(denominator,1):.1f}%</b> of responses</span><span><b>{len(t)}</b> topics</span></div>'
                     f'<h3>Topic distribution within this theme</h3>{_chart(fig,cid)}<h3>Topics and descriptions</h3>')
        for _,row in t.iterrows():
            tid, name = row[topic_col], str(row['topic_name'])
            desc = ''
            if isinstance(mapping,pd.DataFrame) and not mapping.empty and {'theme_name','topic_id'}.issubset(mapping.columns):
                matched = mapping.loc[(mapping['theme_name'].astype(str)==str(theme_name)) & (mapping['topic_id'].astype(str)==str(tid))]
                for field in ('description','topic_description'):
                    if not matched.empty and field in matched.columns and pd.notna(matched.iloc[0][field]):
                        desc = str(matched.iloc[0][field]); break
            subset = group.loc[group[topic_col].astype(str)==str(tid)]
            cols = [c for c in (id_col,response_col) if c and c in subset.columns]
            response_rows = _records(subset[cols].fillna('')) if cols else []
            parts.append(f'<article class="topic"><h4>{html.escape(name)}</h4><p class="muted">Topic {html.escape(str(tid))} · {int(row["Count"]):,} responses · {row["Within Theme (%)"]:.1f}% of this theme</p>'
                         f'<p>{html.escape(desc) if desc else "No additional topic description available."}</p>'
                         f'<details><summary>View responses — {html.escape(name)} ({len(response_rows)})</summary><div class="responses">')
            for r in response_rows:
                text = html.escape(str(r.get(response_col,'')))
                rid = f'<b>{html.escape(str(r.get(id_col)))}</b> — ' if id_col and id_col in r else ''
                parts.append(f'<p>{rid}{text}</p>')
            parts.append('</div></details></article>')
        parts.append('</section>')
    return ''.join(parts)


def _topic_sections(explorer, mapping, denominator, prefix, response_col, id_col):
    """Direct topic findings; no artificial theme headings or groupings."""
    assigned = explorer.loc[explorer['topic_name'].notna() &
                            explorer['topic_name'].astype(str).ne('Unassigned')]
    if assigned.empty:
        return '<p>No named topics were identified.</p>'
    parts = []
    for n, (topic_id, group) in enumerate(assigned.groupby('topic', sort=True), 1):
        name = str(group['topic_name'].iloc[0]); count = len(group)
        matched = (mapping.loc[mapping['topic_id'].astype(str).eq(str(topic_id))]
                   if isinstance(mapping, pd.DataFrame) and not mapping.empty else pd.DataFrame())
        desc = ''
        if not matched.empty:
            for field in ('description', 'topic_description'):
                if field in matched and pd.notna(matched.iloc[0][field]):
                    desc = str(matched.iloc[0][field]); break
        parts.append(f'<section class="theme-card"><p class="eyebrow">TOPIC {n} · BERTopic ID {topic_id}</p>'
                     f'<h2>{html.escape(name)}</h2><div class="metrics"><span><b>{count:,}</b> responses</span>'
                     f'<span><b>{100*count/max(denominator,1):.1f}%</b> of responses</span></div>'
                     f'<p>{html.escape(desc)}</p><details><summary>View responses ({count:,})</summary><div class="responses">')
        for _, record in group.iterrows():
            rid = (f'<b>{html.escape(str(record[id_col]))}</b> — '
                   if id_col and id_col in group.columns else '')
            parts.append(f'<p>{rid}{html.escape(str(record[response_col]))}</p>')
        parts.append('</div></details></section>')
    return ''.join(parts)


def build_interactive_html(results, analysis_type, coded, dictionary, recommendations=None,
                           recommendation_evidence=None, survey_question='', response_column='', id_column=None):
    """Return standalone HTML bytes. No external scripts, server or API required."""
    explorer = results['response_explorer'].copy()
    mapping = results.get('theme_mapping')
    if not isinstance(mapping,pd.DataFrame): mapping = pd.DataFrame()
    total = len(explorer)
    themes_generated = results.get('themes_generated', True)
    if not response_column or response_column not in explorer.columns:
        response_column = results.get('response_column','')
    if not response_column or response_column not in explorer.columns:
        raise ValueError('Original response column is required for the interactive theme evidence.')
    if response_column not in coded.columns:
        raise ValueError('Analysed response column missing from coded data.')
    theme_parts=[]
    if analysis_type == 'sentiment' and 'sentiment' in explorer:
        sentiment_dist = explorer.groupby('sentiment').size().reset_index(name='Count')
        fig = px.pie(sentiment_dist,names='sentiment',values='Count',hole=.35,
                     title='Overall sentiment distribution (%)',color='sentiment',
                     color_discrete_map={'Positive':'#4D9B78','Neutral':'#8794A3','Negative':'#C16B6B'})
        fig.update_traces(textinfo='label+percent',textposition='outside')
        theme_parts.append(f'<h2>Overall sentiment distribution</h2>{_chart(fig,"sentiment-overall")}')
        dist = results.get('sentiment_by_topic',pd.DataFrame())
        if isinstance(dist,pd.DataFrame) and not dist.empty:
            chart = px.bar(dist,x='topic_name',y='Within Topic (%)',color='sentiment',
                           color_discrete_map={'Positive':'#4D9B78','Neutral':'#8794A3','Negative':'#C16B6B'},
                           barmode='stack',hover_data=['Count','Topic Total'],
                           title='Sentiment composition within each topic (%)')
            chart.update_layout(yaxis_range=[0,100],xaxis_tickangle=-35,
                                yaxis_title='Within topic (%)',xaxis_title=None)
            theme_parts.append('<h2>Sentiment by topic</h2><p class="muted">'
                               'Each bar uses responses assigned to that topic as its denominator. '
                               'Short responses were dropped before analysis. '
                               'BERTopic-unassigned responses appear only in the overall distribution.</p>'
                               +_chart(chart,'topic-sentiment-chart'))
        theme_parts.append(('<h2>Topics and themes across all sentiments</h2>'
                            +_theme_sections(explorer,mapping,total,'topic-first',response_column,id_column))
                           if themes_generated else
                           ('<h2>Identified topics</h2><p>Theme grouping skipped: 8 or fewer topics.</p>'
                            +_topic_sections(explorer,mapping,total,'topic-first',response_column,id_column)))
        assigned = explorer.loc[explorer['topic_name'].ne('Unassigned')]
        if not assigned.empty:
            theme_parts.append('<h2>Positive, neutral and negative responses within each topic</h2>')
            for topic_id, rows in assigned.groupby('topic'):
                name = str(rows['topic_name'].iloc[0]); parent = str(rows['theme_name'].iloc[0])
                theme_parts.append('<section class="theme-card"><h3>' + html.escape(name) +
                                   '</h3><p class="muted">'+
                                   (html.escape(parent)+' · ' if themes_generated else '')+
                                   str(len(rows))+' responses</p>')
                for label in SENTIMENTS:
                    subset = rows.loc[rows['sentiment'].eq(label)]
                    pct = 100*len(subset)/max(len(rows),1)
                    theme_parts.append('<details><summary>'+html.escape(label)+': '+str(len(subset))+
                                       ' responses ('+f'{pct:.1f}%'+')</summary><div class="responses">')
                    for _, record in subset.iterrows():
                        response = html.escape(str(record[response_column]))
                        confidence = record.get('confidence_score','')
                        flag = html.escape(str(record.get('review_flag','')))
                        theme_parts.append('<p>'+response+' <small class="muted">(Confidence: '+
                                           html.escape(str(confidence))+'%; '+flag+')</small></p>')
                    theme_parts.append('</div></details>')
                theme_parts.append('</section>')
    else:
        if themes_generated:
            theme_parts.append('<h2>Distribution of findings</h2><div class="chart-pair">'
                               +_chart(_distribution(explorer,'theme_name',total,'Theme distribution (%)'),'overall-themes')
                               +_chart(_topic_chart(explorer,total,'Topic distribution (%)'),'overall-topics')+'</div>')
            theme_parts.append(_theme_sections(explorer,mapping,total,'overall',response_column,id_column))
        else:
            theme_parts.append('<h2>Topic distribution</h2><p>Theme grouping skipped: 8 or fewer topics.</p>'
                               +_chart(_distribution(explorer,'topic_name',total,'Topic distribution (%)'),'overall-topics'))
            theme_parts.append(_topic_sections(explorer,mapping,total,'overall',response_column,id_column))
    evidence_lookup = {t['evidence_id']:t for t in (recommendation_evidence or {}).get('topics',[])}
    rec_parts=[]
    for n,rec in enumerate(recommendations or [],1):
        priority = str(rec.get('priority',''))
        sources=[evidence_lookup[k] for k in rec.get('supporting_evidence_ids',[]) if k in evidence_lookup]
        rec_parts.append(f'<section class="theme-card"><p class="eyebrow">RECOMMENDATION {n} · <span class="priority">{html.escape(priority)}</span></p>'
                         f'<h2>{html.escape(str(rec.get("title","")))}</h2><p>{html.escape(str(rec.get("action","")))}</p>'
                         f'<h3>Evidence and rationale</h3><p>{html.escape(str(rec.get("rationale","")))}</p>'
                         f'<p class="muted">Priority rationale: {html.escape(str(rec.get("priority_reason","")))}</p><h3>Supporting topics</h3>')
        for t in sources:
            parent = (html.escape(str(t['theme_name'])) + ' / ') if t.get('theme_name') else ''
            rec_parts.append(f'<p>{parent}<b>{html.escape(str(t.get("topic_name","")))}</b> '
                             f'(Topic {html.escape(str(t.get("topic_code","")))}; {t.get("response_count",0):,} responses; {t.get("percentage_of_all_responses",0):.1f}% of all responses)</p>')
        rec_parts.append('<details><summary>View supporting original responses</summary><div class="responses">')
        for t in sources:
            rec_parts.append(f'<h4>{html.escape(str(t.get("topic_name","")))}</h4>')
            for sample in t.get('sample_responses',[]): rec_parts.append(f'<p>{html.escape(str(sample))}</p>')
        rec_parts.append('</div></details></section>')
    if not rec_parts: rec_parts=['<p>Recommendations are unavailable for this analysis.</p>']
    # Reuse the same completed report data as the Streamlit Report tab.
    report = compile_report(results, analysis_type, survey_question=survey_question,
                            recommendations=recommendations, recommendation_evidence=recommendation_evidence)
    report_parts = ['<h1>Survey Analysis Report</h1>']
    if report['question']:
        report_parts.append(f'<p><b>Survey question:</b> {html.escape(str(report["question"]))}</p>')
    report_parts.append('<h2>1. Executive Summary</h2>')
    report_parts.append(f'<p>The analysis included {report["total"]:,} responses. '
                        f'{report["assigned"]:,} were assigned to named topics and '
                        f'{report["unassigned"]:,} were unassigned. '
                        f'{len(report["themes"])} {report.get("group_label", "theme")} groups and '
                        f'{len(report["recommendations"])} recommendations were identified.</p>')
    header = 'Theme' if themes_generated else 'Topic'
    report_parts.append('<h2>2. Overview of Findings</h2><div class="table-wrap"><table>'
                        f'<thead><tr><th>{header}</th><th>Sentiment</th><th>Responses</th><th>% of responses</th></tr></thead><tbody>')
    for theme in report['themes']:
        report_parts.append(f'<tr><td>{html.escape(theme["name"])}</td>'
                            f'<td>{html.escape(str(theme["sentiment"] or "—"))}</td>'
                            f'<td>{theme["count"]:,}</td><td>{theme["percentage"]:.1f}%</td></tr>')
    report_parts.append('</tbody></table></div><h2>3. Key Findings</h2>')
    for n, theme in enumerate(report['themes'], 1):
        title = theme['name'] + (f' ({theme["sentiment"]})' if theme['sentiment'] else '')
        report_parts.append(f'<section class="theme-card"><h3>3.{n} {html.escape(title)}</h3>'
                            f'<p class="muted">{theme["count"]:,} responses · {theme["percentage"]:.1f}% of all responses</p>')
        for topic in theme['topics']:
            if themes_generated:
                report_parts.append(f'<h4>{html.escape(topic["name"])}</h4><p>{topic["count"]:,} responses '
                                    f'({topic["percentage"]:.1f}% of all responses)</p>')
            if topic['description']:
                report_parts.append(f'<p>{html.escape(topic["description"])}</p>')
            if analysis_type == 'sentiment':
                breakdown = topic.get('sentiment_counts', {})
                if breakdown:
                    text = '; '.join(f'{label}: {breakdown.get(label,0)} '
                                     f'({100*breakdown.get(label,0)/max(topic["count"],1):.1f}%)'
                                     for label in SENTIMENTS)
                    report_parts.append('<p><b>Sentiment within topic:</b> '+html.escape(text)+'</p>')
                for label, examples in topic.get('examples_by_sentiment',{}).items():
                    if examples:
                        report_parts.append('<details><summary>'+html.escape(label)+' example responses</summary>')
                        for example in examples:
                            report_parts.append('<blockquote>'+html.escape(str(example))+'</blockquote>')
                        report_parts.append('</details>')
        if theme['quotes']:
            report_parts.append('<h4>Illustrative responses</h4>')
            for quote in theme['quotes']:
                report_parts.append(f'<blockquote>{html.escape(quote)}</blockquote>')
        report_parts.append('</section>')
    report_parts.append('<h2>4. Recommendations</h2>')
    if report['recommendations']:
        for n, rec in enumerate(report['recommendations'], 1):
            report_parts.append(f'<section class="theme-card"><h3>4.{n} {html.escape(str(rec["title"]))}</h3>'
                                f'<p><b>Priority:</b> {html.escape(str(rec["priority"]))}</p>'
                                f'<p>{html.escape(str(rec["action"]))}</p>'
                                f'<p><b>Rationale:</b> {html.escape(str(rec["rationale"]))}</p>'
                                f'<p><b>Priority rationale:</b> {html.escape(str(rec["priority_reason"]))}</p>')
            if rec['sources']:
                report_parts.append('<p><b>Supporting topics:</b> ' + '; '.join(
                    html.escape(str(t.get('topic_name',''))) + ' (' + html.escape(str(t.get('theme_name',''))) + ')'
                    for t in rec['sources']) + '</p>')
            report_parts.append('</section>')
    else:
        report_parts.append('<p>Recommendations are unavailable for this analysis.</p>')
    report_parts.append('<p class="muted">Percentages are descriptive and based on analysed responses. '
                        'Model-derived topic assignments may include unassigned responses.</p>')
    data = _json({'coded':_records(coded),'dictionary':_records(dictionary),'responseColumn':response_column})
    safe_question = html.escape(str(survey_question or results.get('survey_question','')))
    doc = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AlgoriThemes — Completed Analysis</title>
<style>
:root{font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:#2c2633;background:#f6f4f9}*{box-sizing:border-box}body{margin:0}header{background:#321b45;color:white;padding:22px max(24px,calc((100vw - 1320px)/2))}header h1{margin:0 0 6px;font-size:24px}header p{margin:0;color:#e9dff1}.shell{max-width:1320px;margin:auto;padding:20px 24px 60px}nav{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:22px}nav button,.button{cursor:pointer;border:1px solid #d7cce2;background:white;color:#321b45;padding:11px 16px;border-radius:9px;font-weight:650}nav button.active,.button.primary{background:#321b45;color:white}.page{display:none}.page.active{display:block}.theme-card,.panel{background:white;border:1px solid #e5dfeb;border-radius:14px;padding:24px;margin:18px 0;box-shadow:0 2px 8px #1a122008}.topic{border-top:1px solid #eee8f1;padding:14px 0}.topic h4{font-size:18px;margin:6px 0}.muted,.eyebrow{color:#6f6678}.eyebrow{font-size:12px;font-weight:750;letter-spacing:.08em}.metrics{display:flex;gap:14px;flex-wrap:wrap;margin:18px 0}.metrics span{padding:14px;background:#f5f0fa;border-radius:10px}.chart-pair{display:grid;grid-template-columns:minmax(0,1fr);gap:20px}.chart{width:100%;min-width:0;overflow:visible}.chart-pair .chart{background:white;border-radius:12px;border:1px solid #e5dfeb}details{margin-top:12px;background:#f8f6fa;border-radius:9px;padding:12px}summary{cursor:pointer;color:#432b60;font-weight:650}.responses{max-height:450px;overflow:auto}.responses p{border-bottom:1px solid #e8e2ec;padding-bottom:9px;white-space:pre-wrap;overflow-wrap:anywhere}.filters{display:flex;flex-wrap:wrap;gap:12px;margin:18px 0}.filters label{font-size:13px;font-weight:650;display:grid;gap:5px}.filters input,.filters select{min-width:180px;border:1px solid #d5cbdc;padding:10px;border-radius:8px;background:white}.table-wrap{overflow:auto;max-height:650px;border:1px solid #e1dbe6;border-radius:10px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;border-bottom:1px solid #eee9f2;padding:10px;vertical-align:top;min-width:90px;max-width:400px;overflow-wrap:anywhere}th{position:sticky;top:0;background:#eee7f4;z-index:1}tr:nth-child(even){background:#fbf9fd}.pagination{display:flex;gap:12px;align-items:center;justify-content:flex-end;margin:12px 0}.pagination button{padding:8px 14px;cursor:pointer}.priority{color:#8a3c13}.sentiment{margin:26px 0}h2{color:#321b45}h3{margin-top:22px}p{line-height:1.55}@media(max-width:900px){.chart-pair{grid-template-columns:1fr}.shell{padding:16px}.theme-card{padding:16px}}@media print{nav,.filters,.pagination,.button{display:none!important}.page{display:block!important}.theme-card{break-inside:avoid}.chart-pair{display:block}}
</style></head><body><header><h1>AlgoriThemes</h1><p>Completed Survey Analysis · __QUESTION__</p></header><main class="shell"><nav><button class="active" data-tab="themes">Themes</button><button data-tab="explorer">Response Explorer</button><button data-tab="recommendations">Recommendations</button><button data-tab="report">Report</button></nav>
<section class="page active" id="page-themes"><h1>Thematic Analysis</h1><p class="muted">__TOTAL__ analysed responses · Complete charts and theme/topic findings</p>__THEMES__</section>
<section class="page" id="page-explorer"><h1>Response Explorer</h1><p>Search and filter the complete coded survey responses.__REVIEW_NOTE__</p><div class="filters"><label>Sentiment<select id="filter-sentiment"><option value="">All sentiments</option></select></label><label>Review flag<select id="filter-review"><option value="">All review statuses</option></select></label><label>Theme<select id="filter-theme"><option value="">All themes</option></select></label><label>Topic<select id="filter-topic"><option value="">All topics</option></select></label><label>Search original responses<input id="filter-search" placeholder="Type a word or phrase"></label></div><p id="match-count"></p><div class="table-wrap"><table id="coded-table"></table></div><div class="pagination"><button id="prev">Previous</button><span id="page-status"></span><button id="next">Next</button></div><button class="button primary" id="download-coded">Download coded responses CSV</button><h2>Coding Dictionary</h2><div class="table-wrap"><table id="dictionary-table"></table></div><p><button class="button" id="download-dictionary">Download coding dictionary CSV</button></p></section>
<section class="page" id="page-recommendations"><h1>Recommendations</h1>__RECOMMENDATIONS__</section>
<section class="page" id="page-report">__REPORT__</section></main><script>__PLOTLY__</script><script type="application/json" id="analysis-data">__DATA__</script>
<script>
(()=>{'use strict';const d=JSON.parse(document.getElementById('analysis-data').textContent),rows=d.coded,dict=d.dictionary;let filtered=rows,page=0;const limit=75,$=id=>document.getElementById(id);
function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function options(el,values){const keep=el.value;el.innerHTML='<option value="">All '+(el.id==='filter-theme'?'themes':el.id==='filter-topic'?'topics':'sentiments')+'</option>'+Array.from(new Set(values.filter(x=>x!==null&&x!==undefined&&x!==''))).sort().map(v=>'<option value="'+esc(v)+'">'+esc(v)+'</option>').join('');if(Array.from(el.options).some(o=>o.value===keep))el.value=keep;}
function table(el,data){const keys=data.length?Object.keys(data[0]):[];el.innerHTML='<thead><tr>'+keys.map(k=>'<th>'+esc(k)+'</th>').join('')+'</tr></thead><tbody>'+data.map(r=>'<tr>'+keys.map(k=>'<td>'+esc(r[k])+'</td>').join('')+'</tr>').join('')+'</tbody>';}
const s=$('filter-sentiment'),r=$('filter-review'),t=$('filter-theme'),p=$('filter-topic'),q=$('filter-search');if(!rows.some(row=>'Sentiment'in row))s.parentElement.style.display='none';if(!rows.some(row=>'Review Flag'in row))r.parentElement.style.display='none';options(s,rows.map(row=>row.Sentiment));options(r,rows.map(row=>row['Review Flag']));
function filter(){const sentiment=s.value,theme=t.value,topic=p.value,search=q.value.toLocaleLowerCase();const first=rows.filter(row=>(!sentiment||row.Sentiment===sentiment)&&(!r.value||row['Review Flag']===r.value));options(t,first.map(r=>r['Theme Name']));const second=first.filter(r=>!t.value||r['Theme Name']===t.value);options(p,second.map(r=>r['Topic Name']));filtered=second.filter(r=>(!p.value||r['Topic Name']===p.value)&&(!search||String(r[d.responseColumn]??'').toLocaleLowerCase().includes(search)));page=0;render();}
function render(){const total=Math.max(1,Math.ceil(filtered.length/limit));page=Math.min(page,total-1);const chunk=filtered.slice(page*limit,(page+1)*limit);const keys=rows.length?Object.keys(rows[0]):[];const tableEl=$('coded-table');tableEl.innerHTML='<thead><tr>'+keys.map(k=>'<th>'+esc(k)+'</th>').join('')+'</tr></thead><tbody>'+chunk.map(r=>'<tr>'+keys.map(k=>'<td>'+esc(r[k])+'</td>').join('')+'</tr>').join('')+'</tbody>';$('match-count').textContent=filtered.length.toLocaleString()+' matching responses';$('page-status').textContent='Page '+(page+1)+' of '+total;$('prev').disabled=page===0;$('next').disabled=page>=total-1;}
[s,r,t,p,q].forEach(el=>el.addEventListener(el===q?'input':'change',filter));$('prev').onclick=()=>{page--;render()};$('next').onclick=()=>{page++;render()};table($('dictionary-table'),dict);filter();
function csv(data){if(!data.length)return '';const keys=Object.keys(data[0]);return [keys,...data.map(r=>keys.map(k=>r[k]))].map(row=>row.map(v=>'"'+String(v??'').replace(/"/g,'""')+'"').join(',')).join('\r\n');}
function download(data,name){const blob=new Blob(['\ufeff'+csv(data)],{type:'text/csv;charset=utf-8'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
$('download-coded').onclick=()=>download(rows,'coded_responses.csv');$('download-dictionary').onclick=()=>download(dict,'coding_dictionary.csv');
let chartsRendered=false;function draw(){if(chartsRendered)return;chartsRendered=true;document.querySelectorAll('.chart').forEach(el=>{const node=$('data-'+el.id);if(!node)return;const spec=JSON.parse(node.textContent);try{spec.layout=spec.layout||{};spec.layout.dragmode=false;for(const axis of ['xaxis','yaxis']){if(spec.layout[axis])spec.layout[axis].fixedrange=true;}Plotly.newPlot(el,spec.data,spec.layout,{responsive:true,displaylogo:false,displayModeBar:false,editable:false,edits:{annotationPosition:false,annotationTail:false,annotationText:false,axisTitleText:false,colorbarPosition:false,colorbarTitleText:false,legendPosition:false,legendText:false,shapePosition:false,titleText:false},scrollZoom:false,doubleClick:false});}catch(err){console.error('Chart failed:',el.id,err);el.textContent='Chart could not be displayed.';}});}
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{document.querySelectorAll('nav button').forEach(x=>x.classList.toggle('active',x===b));document.querySelectorAll('.page').forEach(x=>x.classList.toggle('active',x.id==='page-'+b.dataset.tab));if(b.dataset.tab==='themes')setTimeout(()=>{document.querySelectorAll('.js-plotly-plot').forEach(el=>{try{Plotly.Plots.resize(el)}catch(err){console.warn(err)}})},80)});draw();
})();
</script></body></html>'''
    review_note = (
        " Confidence scores are model probabilities, not validated accuracy "
        f"estimates; Review flags mark predictions below "
        f"{100 * results.get('sentiment_confidence_threshold', 0.70):g}% confidence."
        if analysis_type == 'sentiment' else ''
    )
    if not themes_generated:
        doc = doc.replace('data-tab="themes">Themes', 'data-tab="themes">Topics')
        doc = doc.replace('<h1>Thematic Analysis</h1>', '<h1>Topic Analysis</h1>')
        doc = doc.replace('Complete charts and theme/topic findings', 'Complete charts and topic findings')
        doc = doc.replace('<label>Theme<select id="filter-theme"><option value="">All themes</option></select></label>',
                          '<label style="display:none">Theme<select id="filter-theme"><option value="">All themes</option></select></label>')
    return (doc.replace('__REVIEW_NOTE__', review_note)
            .replace('__QUESTION__',safe_question).replace('__TOTAL__',f'{total:,}')
            .replace('__THEMES__',''.join(theme_parts)).replace('__RECOMMENDATIONS__',''.join(rec_parts)).replace('__REPORT__',''.join(report_parts))
            .replace('__PLOTLY__',get_plotlyjs()).replace('__DATA__',data)).encode('utf-8')
