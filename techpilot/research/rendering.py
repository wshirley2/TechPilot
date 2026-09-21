"""Readable Markdown over validated facts; short citations never replace IDs."""

from urllib.parse import quote, urlsplit

from .contracts import EvidenceFragment, ReportArtifact, ReportClaim, ResearchTask, _literal


def reference_numbers(task: ResearchTask, report: ReportArtifact) -> dict[str, int]:
    """First visible use: summary, then matrix rows in declared candidate order."""
    ordered = [claim for claim in report.claims if claim.candidate is None and claim.conclusion is not None]
    cells = {(claim.topic, claim.candidate): claim for claim in report.claims if claim.candidate is not None}
    for topic in dict.fromkeys(topic for topic, _ in cells):
        ordered.extend(cells[(topic, candidate)] for candidate in task.candidates)
    refs = dict.fromkeys(ref for claim in ordered for ref in claim.evidence_ids)
    return {ref: number for number, ref in enumerate(refs, 1)}


def _statement(claim: ReportClaim, numbers: dict[str, int]) -> str:
    if claim.conclusion is None:
        return "未知/证据不足"
    refs = " ".join(f"[{numbers[ref]}](#ref-{numbers[ref]})" for ref in claim.evidence_ids)
    return _literal(claim.conclusion) + (f" {refs}" if refs else "")


def _source_link(source_id: str) -> str:
    """Only emit safe absolute web links; arbitrary source IDs remain literal."""
    try:
        url = urlsplit(source_id)
        if (url.scheme in {"http", "https"} and url.hostname and not url.username and not url.password
                and not any(ord(char) < 32 for char in source_id)):
            target = quote(source_id, safe=":/?#=&%+@-._~")
            return f"[查看来源文档]({target})"
    except ValueError:
        pass
    return _literal(source_id)


def render_report(task: ResearchTask, report: ReportArtifact, evidence: tuple[EvidenceFragment, ...]) -> str:
    """Render only after validate_report; does not perform I/O or grant delivery."""
    numbers = reference_numbers(task, report)
    evidence_map = {fragment.evidence_id: fragment for fragment in evidence}
    lines = [f"# {_literal(report.title)}", "", "## 问题与约束", "", _literal(task.question), "",
             "候选方案：" + "、".join(_literal(candidate) for candidate in task.candidates) + "。", ""]
    lines.extend(f"- {_literal(constraint)}" for constraint in task.constraints)
    lines.extend(["", "## 结论摘要", ""])
    summary = [claim for claim in report.claims if claim.candidate is None and claim.conclusion is not None]
    if summary:
        lines.extend(f"- **{_literal(claim.topic)}**：{_statement(claim, numbers)}" for claim in summary)
    else:
        lines.append("未提供独立的综合选型结论；请结合下列事实与未知项判断。")

    comparison = [claim for claim in report.claims if claim.candidate is not None]
    if comparison:
        lines.extend(["", "## 同维度比较", "",
                      "| 比较维度 | " + " | ".join(_literal(candidate) for candidate in task.candidates) + " |",
                      "| --- | " + " | ".join("---" for _ in task.candidates) + " |"])
        cells = {(claim.topic, claim.candidate): claim for claim in comparison}
        for topic in dict.fromkeys(claim.topic for claim in comparison):
            values = [_statement(cells[(topic, candidate)], numbers) for candidate in task.candidates]
            lines.append("| " + _literal(topic) + " | " + " | ".join(values) + " |")
        lines.extend(["", "说明：这里比较资料已证实的能力；未写出不代表不支持，也不构成完整功能清单。"])

    lines.extend(["", "## 未知与待确认事项", ""])
    unknowns = [claim for claim in report.claims if claim.conclusion is None]
    if unknowns:
        for claim in unknowns:
            label = f"{claim.candidate} · {claim.topic}" if claim.candidate else claim.topic
            lines.append(f"- {_literal(label)}：未知/证据不足。")
    else:
        lines.append("本报告未列出未知项；这不代表已覆盖所有选型维度。")

    receipt_name = quote(report.output_path.with_suffix(".json").name, safe="")
    lines.extend(["", "## 资料与引用", "",
                  "正文编号可跳转到下方摘录。同一片段重复使用时沿用同一编号。",
                  "以下为本地快照中的逐字摘录；中文整理稿不是上游官方原文或官方译文。", "",
                  f"[完整校验记录]({receipt_name}) 保存引用编号、完整证据 ID、快照版本、摘要和快照文件路径。",
                  "编号只在本报告内有效，不替代机器校验使用的稳定 ID。", ""])
    for ref, number in numbers.items():
        fragment = evidence_map[ref]
        candidates = dict.fromkeys(claim.candidate for claim in report.claims
                                   if claim.candidate is not None and ref in claim.evidence_ids)
        label = " · " + _literal("、".join(candidates)) if candidates else ""
        lines.extend([f'<a id="ref-{number}"></a>', "", f"### 引用 {number}{label}", "",
                      f"{_source_link(fragment.source_id)} · 本地快照第 {fragment.start_line}–{fragment.end_line} 行", ""])
        lines.extend("> " + _literal(line) for line in fragment.quote.splitlines())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
