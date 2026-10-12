from cli_data_tool.exporter import export_text


def test_export_text_joins_items_with_newlines():
    assert export_text(["one", "two"]) == "one\ntwo"
