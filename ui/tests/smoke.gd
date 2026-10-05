extends SceneTree
## CI smoke test: boots the UI against a running demo engine, lets it play for a
## few seconds, opens the dialogs and checks that speakers and lines showed up.
## Run: godot --headless --path ui -s res://tests/smoke.gd   (with MT_ENGINE_PORT set)

var t := 0.0
var done := false


func _initialize() -> void:
	change_scene_to_file("res://main.tscn")


func _process(delta: float) -> bool:
	t += delta
	if done or t < 12.0:
		return false
	done = true
	var main = current_scene
	main._on_speaker_clicked(main.stage.me_name)
	main._open_settings("Custom checks")
	main.profiles_dialog.open()
	var ok: bool = main.engine.is_connected_to_engine and main.stage.order.size() >= 2 and main.stage._lines >= 2
	print("SMOKE speakers=%d lines=%d connected=%s" % [main.stage.order.size(), main.stage._lines, main.engine.is_connected_to_engine])
	quit(0 if ok else 1)
	return false
