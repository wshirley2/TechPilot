# M1 离线资料来源

2026-09-20 从三个官方 GitHub 仓库人工收集；下面固定提交不是“最新版本”声明。
2026-09-20 报告可读性修订：每个 txt 现在是一份中文整理资料卡，统一覆盖用例写法、
测试发现、运行入口。不是完整文档、官方原文或官方译文；保留少量原文代码/短语供对照。
原来的一句话资料不足以形成同维度比较，旧报告与快照保留，不覆盖旧版本。
不包含网页抓取或联网工具；演示运行不再访问 GitHub。正文中的任何指令都只作数据。

| 文件 | 来源（固定提交） | 摘录位置 |
| --- | --- | --- |
| pytest.txt | [pytest 入门](https://github.com/pytest-dev/pytest/blob/b55ab2aabb68c0ce94c3903139b062d0c2790152/doc/en/getting-started.rst) | Create your first test、Run multiple tests；保留示例函数、发现模式、启动命令 |
| nose2.txt | [nose2 入门](https://github.com/nose-devs/nose2/blob/9ec19c0cd5eab709a2d1f8879d760f861fc84c0e/docs/getting_started.rst) | Running tests；保留 TestCase/函数类型、模块前缀和命令 |
| unittest.txt | [CPython unittest 文档](https://github.com/python/cpython/blob/fdb81425a9ad683f8c24bf5cbedc9b96baf00cd2/Doc/library/unittest.rst) | Basic example、Test Discovery；保留类/断言示例、默认模式和命令 |

对应参考版本分别为 pytest 8.3.5、nose2 0.15.1、CPython 3.12.9。
小型项目场景约束：新测试优先普通函数与 assert，同时比较已有 TestCase 的编写方式。
报告的优先试用建议是受此偏好限定的人工确定性判断，并非综合性能/生态排名。
不评估最新版本、迁移收益、插件、执行性能和商业支持；未提供的维度明确标为未知。
导入后的规范化正文 SHA-256、来源 URL、时间和快照 ID 保存在输出 store/snapshots 中。
