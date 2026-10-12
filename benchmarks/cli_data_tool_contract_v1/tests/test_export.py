from cli_data_tool.cli import DEFAULT_ITEMS, main


def test_export_uses_text_output_by_default(capsys):
    assert main(["export"]) == 0

    captured = capsys.readouterr()
    assert captured.out == "\n".join(DEFAULT_ITEMS) + "\n"


def test_export_accepts_custom_items(capsys):
    assert main(["export", "--items", "one", "two"]) == 0

    captured = capsys.readouterr()
    assert captured.out == "one\ntwo\n"
