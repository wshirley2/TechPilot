"""Run an explicit, offline M1 example with fixed responses and bounded writes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from techpilot.engine.permissions import PermissionDecision, PermissionEffect
from techpilot.research.contracts import ReportArtifact, ReportClaim, ResearchTask
from techpilot.research.runtime_files import RuntimeResearchFiles
from techpilot.research.workflow import ResearchWorkflow

SOURCES = (
    ("pytest", "https://github.com/pytest-dev/pytest/blob/b55ab2aabb68c0ce94c3903139b062d0c2790152/doc/en/getting-started.rst",
     ("普通测试函数 + assert；入门示例不继承 TestCase。",
      "当前目录及子目录中的 test_*.py 或 *_test.py。", "pytest")),
    ("nose2", "https://github.com/nose-devs/nose2/blob/9ec19c0cd5eab709a2d1f8879d760f861fc84c0e/docs/getting_started.rst",
     ("可加载 TestCase 子类，也可加载 test 开头的函数。",
      "模块名以 test 开头；查找工作目录中的包及 test 开头的子目录。", "nose2")),
    ("unittest", "https://github.com/python/cpython/blob/fdb81425a9ad683f8c24bf5cbedc9b96baf00cd2/Doc/library/unittest.rst",
     ("基础示例继承 TestCase，使用 assertEqual 等断言方法。",
      "discover 默认匹配 test*.py；模块或包必须可从项目顶层导入。", "python -m unittest discover")),
)
DIMENSIONS = ("用例写法", "测试发现", "运行入口")


class DemoWritePrompt:
    """Explicit demo-only preapproval; never installed into ordinary Chat."""

    def __init__(self, repository: Path, roots: tuple[Path, ...]):
        self.repository = repository.resolve()
        self.roots = tuple(root.resolve() for root in roots)

    def decide(self, request):
        path = (self.repository / request.normalized_arguments.get("file_path", "")).resolve()
        if (request.tool_name == "write_file" and request.effect is PermissionEffect.WRITE
                and any(path.is_relative_to(root) and path != root for root in self.roots)
                and not path.exists() and request.trusted_preview is not None):
            return PermissionDecision.allow("Explicit M1 demo: create only within its artifact/store directories")
        return PermissionDecision.deny("Outside the explicit M1 demo write scope")


def run_demo(output: Path):
    output = output.resolve()
    if not output.is_relative_to(ROOT) or output == ROOT or output.exists():
        raise ValueError("Choose a new output directory inside this repository")
    task = ResearchTask("m1-python-test-tools", "小型 Python 项目应先试用哪种测试框架？已有 TestCase 用例时如何看待候选方案？",
                        ("pytest", "nose2", "unittest"),
                        ("新测试优先普通函数与 assert 写法；不要求本轮完成旧用例迁移。",
                         "只比较用例写法、测试发现、运行入口；不作完整功能或性能排名。",
                         "资料固定在 pytest 8.3.5、nose2 0.15.1、CPython 3.12.9，不代表最新版本。",
                         "资料卡是依据官方固定文档的中文整理，非官方译文；报告是确定性人工示例，不是模型自动调研。"),
                        output / "reports")
    files = RuntimeResearchFiles(ROOT, output / "sessions",
                                 DemoWritePrompt(ROOT, (output / "store", task.artifact_directory)))
    workflow = ResearchWorkflow(task, output / "store", files)
    workflow.initialize()
    snapshots, evidence, comparisons = [], [], []
    writing_refs = {}
    for name, url, conclusions in SOURCES:
        snapshot = workflow.import_source(url, ROOT / "tests" / "fixtures" / "research" / f"{name}.txt")
        snapshots.append(snapshot.snapshot_id)
        for dimension, conclusion in zip(DIMENSIONS, conclusions):
            fragment, = workflow.query(snapshot.snapshot_id, dimension + "：")
            evidence.append(fragment)
            comparisons.append(ReportClaim(dimension, conclusion, (fragment.evidence_id,), candidate=name))
            if dimension == "用例写法":
                writing_refs[name] = fragment.evidence_id
    claims = [
        ReportClaim("新项目先试用 pytest",
                    "在本例偏好普通函数与 assert 的前提下，pytest 入门写法直接符合这一要求，建议先做小规模试用。"
                    "这是基于写法偏好的有限判断，不是综合排名。", (writing_refs["pytest"],)),
        ReportClaim("已有 TestCase 时先核对迁移需求",
                    "nose2 文档明确加载 TestCase 子类及测试函数；unittest 的基础示例也基于 TestCase。"
                    "这说明存在可比较的用例组织方式，但不足以证明迁移值得或能无缝完成。",
                    (writing_refs["nose2"], writing_refs["unittest"])),
        *comparisons,
        ReportClaim("项目现有用例是否完全兼容、迁移成本多高——需用实际用例试跑", None),
        ReportClaim("插件适配和执行性能——需补充目标插件资料与同条件测量", None),
        ReportClaim("商业支持与价格——需另查对应服务说明", None),
    ]
    report = ReportArtifact(task.task_id, tuple(claims), tuple(snapshots), task.artifact_directory / "comparison.md",
                            title="Python 测试框架选型比较（固定资料示例）")
    return workflow.deliver(report, tuple(evidence))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".tmp" / f"m1-demo-{uuid4().hex[:10]}")
    args = parser.parse_args()
    delivery = run_demo(args.output_dir)
    print(f"Delivered: {delivery.report_path}")
    print(f"Receipt: {delivery.receipt_path}")
    print("Fixed offline transcript; no network provider or model tokens used.")
