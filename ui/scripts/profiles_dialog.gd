class_name ProfilesDialog
extends Window
## Everyone the app has met, with stats built up across meetings.

signal delete_requested(name: String)
signal rename_requested(old_name: String, new_name: String)

const COLUMNS := ["Name", "Meetings", "Talk (min)", "Avg share", "Pace (wpm)", "Questions / mtg", "Usual mood", "Topics"]

var _tree: Tree
var _rename: LineEdit
var _profiles := {}
var _picker: FileDialog
var _picture_for := ""


func _ready() -> void:
	title = "Speaker profiles"
	size = Vector2i(980, 560)
	min_size = Vector2i(700, 400)
	visible = false
	close_requested.connect(hide)

	var bg := PanelContainer.new()
	bg.add_theme_stylebox_override("panel", Palette.panel_style(Palette.BG, 0, 12))
	bg.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	add_child(bg)
	var root := VBoxContainer.new()
	bg.add_child(root)

	_tree = Tree.new()
	_tree.columns = COLUMNS.size()
	_tree.column_titles_visible = true
	_tree.hide_root = true
	_tree.select_mode = Tree.SELECT_ROW
	_tree.size_flags_vertical = Control.SIZE_EXPAND_FILL
	for i in COLUMNS.size():
		_tree.set_column_title(i, COLUMNS[i])
		_tree.set_column_expand(i, i == 0 or i == COLUMNS.size() - 1)
		_tree.set_column_custom_minimum_width(i, 150 if i == 0 else 90)
	root.add_child(_tree)

	var row := HBoxContainer.new()
	_rename = LineEdit.new()
	_rename.placeholder_text = "New name (an existing name merges the two)"
	_rename.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	row.add_child(_rename)
	var rename_btn := Button.new()
	rename_btn.text = "Rename selected"
	rename_btn.pressed.connect(func():
		var name := _selected_name()
		var new_name := _rename.text.strip_edges()
		if name != "" and new_name != "":
			rename_requested.emit(name, new_name)
			_rename.text = "")
	row.add_child(rename_btn)
	var picture_btn := Button.new()
	picture_btn.text = "Set picture..."
	picture_btn.tooltip_text = "Pick a photo for the selected person. It shows inside their bubble."
	picture_btn.pressed.connect(_pick_picture)
	row.add_child(picture_btn)
	var no_picture_btn := Button.new()
	no_picture_btn.text = "Remove picture"
	no_picture_btn.pressed.connect(func():
		var name := _selected_name()
		if name != "":
			Avatars.remove(name)
			set_profiles(_profiles))
	row.add_child(no_picture_btn)
	var delete_btn := Button.new()
	delete_btn.text = "Delete selected"
	delete_btn.pressed.connect(func():
		var name := _selected_name()
		if name != "":
			delete_requested.emit(name))
	row.add_child(delete_btn)
	var close := Button.new()
	close.text = "Close"
	close.pressed.connect(hide)
	row.add_child(close)
	root.add_child(row)

	_picker = FileDialog.new()
	_picker.file_mode = FileDialog.FILE_MODE_OPEN_FILE
	_picker.access = FileDialog.ACCESS_FILESYSTEM
	_picker.use_native_dialog = true
	_picker.filters = PackedStringArray(["*.png, *.jpg, *.jpeg, *.webp ; Pictures"])
	_picker.title = "Pick a profile picture"
	_picker.file_selected.connect(_on_picture_selected)
	add_child(_picker)


func _pick_picture() -> void:
	_picture_for = _selected_name()
	if _picture_for == "":
		return
	_picker.popup_centered_ratio(0.6)


func _on_picture_selected(path: String) -> void:
	if _picture_for == "":
		return
	var err := Avatars.set_from_file(_picture_for, path)
	if err != "":
		var dlg := AcceptDialog.new()
		dlg.title = "Profile picture"
		dlg.dialog_text = err
		add_child(dlg)
		dlg.confirmed.connect(dlg.queue_free)
		dlg.popup_centered()
	set_profiles(_profiles)


func _selected_name() -> String:
	var item := _tree.get_selected()
	return item.get_metadata(0) if item else ""


func set_profiles(profiles: Dictionary) -> void:
	_profiles = profiles
	if _tree == null:
		return
	_tree.clear()
	var root := _tree.create_item()
	var names := profiles.keys()
	names.sort_custom(func(a, b): return float(profiles[a].get("talk_minutes", 0)) > float(profiles[b].get("talk_minutes", 0)))
	for name in names:
		var p: Dictionary = profiles[name]
		var item := _tree.create_item(root)
		item.set_metadata(0, name)
		item.set_text(0, name + ("  (you)" if p.get("is_me", false) else ""))
		item.set_custom_color(0, Palette.speaker_color(name, p.get("is_me", false)))
		var avatar := Avatars.texture(name)
		if avatar != null:
			item.set_icon(0, avatar)
			item.set_icon_max_width(0, 28)
		item.set_text(1, str(int(p.get("meetings", 0))))
		item.set_text(2, "%.1f" % float(p.get("talk_minutes", 0.0)))
		item.set_text(3, Palette.percent(float(p.get("avg_talk_share", 0.0))))
		item.set_text(4, str(int(p.get("wpm", 0))))
		item.set_text(5, str(p.get("questions_per_meeting", 0)))
		var mood = p.get("top_mood")
		item.set_text(6, "%s %s" % [Palette.mood_emoji(mood), mood] if mood != null else "-")
		if mood != null:
			item.set_custom_color(6, Palette.mood_color(mood))
		item.set_text(7, ", ".join(p.get("top_topics", [])))


func open() -> void:
	set_profiles(_profiles)
	popup_centered()
