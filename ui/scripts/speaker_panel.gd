class_name SpeakerPanel
extends VBoxContainer
## Sidebar card for the selected speaker: live meeting stats, mood over time,
## check hits and the long-term profile built up across meetings.

signal rename_requested(old_name: String, new_name: String)


class Sparkline:
	extends Control
	var points: Array = []  # valence values -1..1

	func _draw() -> void:
		var r := Rect2(Vector2.ZERO, size)
		draw_rect(r, Color(1, 1, 1, 0.03))
		var mid := size.y * 0.5
		draw_line(Vector2(0, mid), Vector2(size.x, mid), Color(1, 1, 1, 0.08), 1.0)
		if points.size() < 2:
			draw_string(get_theme_default_font(), Vector2(8, mid + 5), "mood over time shows up here",
				HORIZONTAL_ALIGNMENT_LEFT, -1, 12, Palette.TEXT_DIM)
			return
		var pts := PackedVector2Array()
		for i in points.size():
			var x := size.x * float(i) / float(points.size() - 1)
			var y := mid - float(points[i]) * (size.y * 0.42)
			pts.append(Vector2(x, y))
		draw_polyline(pts, Palette.ACCENT, 2.0, true)
		for i in pts.size():
			var v := float(points[i])
			draw_circle(pts[i], 3.0, Palette.GOOD if v > 0.15 else (Palette.BAD if v < -0.15 else Palette.TEXT_DIM))


var speaker_name := ""
var _title: Label
var _rename_edit: LineEdit
var _grid: GridContainer
var _topics: Label
var _mood_label: Label
var _spark: Sparkline
var _hits: Label
var _profile: Label
var _placeholder: Label


func _ready() -> void:
	add_theme_constant_override("separation", 8)
	_placeholder = Label.new()
	_placeholder.text = "Click a speaker in the circle to see their stats and profile."
	_placeholder.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_placeholder.add_theme_color_override("font_color", Palette.TEXT_DIM)
	add_child(_placeholder)

	_title = Label.new()
	_title.add_theme_font_size_override("font_size", 22)
	add_child(_title)

	var rename_row := HBoxContainer.new()
	_rename_edit = LineEdit.new()
	_rename_edit.placeholder_text = "Rename / merge into..."
	_rename_edit.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_rename_edit.text_submitted.connect(func(_t): _do_rename())
	rename_row.add_child(_rename_edit)
	var rename_btn := Button.new()
	rename_btn.text = "Rename"
	rename_btn.pressed.connect(_do_rename)
	rename_row.add_child(rename_btn)
	add_child(rename_row)

	add_child(_section("This meeting"))
	_grid = GridContainer.new()
	_grid.columns = 2
	_grid.add_theme_constant_override("h_separation", 16)
	add_child(_grid)

	_topics = Label.new()
	_topics.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_topics.add_theme_color_override("font_color", Palette.TEXT_DIM)
	add_child(_topics)

	add_child(_section("Mood"))
	_mood_label = Label.new()
	_mood_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	add_child(_mood_label)
	_spark = Sparkline.new()
	_spark.custom_minimum_size = Vector2(0, 70)
	add_child(_spark)

	add_child(_section("Checks"))
	_hits = Label.new()
	_hits.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	add_child(_hits)

	add_child(_section("Profile (all meetings)"))
	_profile = Label.new()
	_profile.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	add_child(_profile)
	show_speaker("", {}, {}, {})


func _section(text: String) -> Label:
	var l := Label.new()
	l.text = text.to_upper()
	l.add_theme_font_size_override("font_size", 12)
	l.add_theme_color_override("font_color", Palette.ACCENT)
	return l


func _do_rename() -> void:
	var new_name := _rename_edit.text.strip_edges()
	if speaker_name != "" and new_name != "" and new_name != speaker_name:
		rename_requested.emit(speaker_name, new_name)
		_rename_edit.text = ""


func show_speaker(name: String, live: Dictionary, stats: Dictionary, profile: Dictionary) -> void:
	speaker_name = name
	var has := name != ""
	_placeholder.visible = not has
	for child in get_children():
		if child != _placeholder:
			child.visible = has
	if not has:
		return
	_title.text = name
	_title.add_theme_color_override("font_color", live.get("color", Palette.TEXT))

	for child in _grid.get_children():
		child.queue_free()
	var rows := [
		["Talk time", Palette.duration(float(stats.get("talk_seconds", 0.0)))],
		["Share of talk", Palette.percent(float(stats.get("talk_share", 0.0)))],
		["Turns", str(int(stats.get("utterances", 0)))],
		["Words", str(int(stats.get("words", 0)))],
		["Pace", "%d wpm" % int(stats.get("wpm", 0))],
		["Questions", str(int(stats.get("questions", 0)))],
		["Interruptions", str(int(stats.get("interruptions", 0)))],
		["Filler words", str(int(stats.get("fillers", 0)))],
		["Longest turn", Palette.duration(float(stats.get("longest_turn_seconds", 0.0)))],
	]
	for row in rows:
		var k := Label.new()
		k.text = row[0]
		k.add_theme_color_override("font_color", Palette.TEXT_DIM)
		_grid.add_child(k)
		var v := Label.new()
		v.text = row[1]
		_grid.add_child(v)

	var topics: Array = stats.get("top_topics", [])
	_topics.text = ("Talks about: " + ", ".join(topics)) if not topics.is_empty() else ""

	var mood = stats.get("mood")
	if mood == null:
		mood = live.get("mood")
	if mood == null:
		_mood_label.text = "No mood yet (turn on Local AI in Settings)."
		_mood_label.add_theme_color_override("font_color", Palette.TEXT_DIM)
	else:
		var counts: Dictionary = stats.get("mood_counts", {})
		var parts: Array = []
		for m in counts:
			parts.append("%s %d" % [m, int(counts[m])])
		var reason: String = live.get("mood_reason", "")
		_mood_label.text = "Now: %s%s\n%s" % [str(mood), (" (" + reason + ")") if reason != "" else "", ", ".join(parts)]
		_mood_label.add_theme_color_override("font_color", Palette.mood_color(mood))
	var vals: Array = []
	for m in stats.get("mood_history", []):
		vals.append(float(m.get("valence", 0.0)))
	_spark.points = vals
	_spark.queue_redraw()

	var hit_parts: Array = []
	var hits: Dictionary = stats.get("check_hits", {})
	for h in hits:
		hit_parts.append("%s: %d" % [h, int(hits[h])])
	var facts: Dictionary = stats.get("fact_checks", {})
	for f in facts:
		hit_parts.append("facts %s: %d" % [f, int(facts[f])])
	_hits.text = "\n".join(hit_parts) if not hit_parts.is_empty() else "Nothing flagged yet."

	if profile.is_empty():
		_profile.text = "First meeting with %s. A profile is saved when you press Stop." % name
	else:
		var lines: Array = [
			"Meetings: %d   Talk: %.1f min" % [int(profile.get("meetings", 0)), float(profile.get("talk_minutes", 0.0))],
			"Avg share: %s   Pace: %d wpm" % [Palette.percent(float(profile.get("avg_talk_share", 0.0))), int(profile.get("wpm", 0))],
			"Questions per meeting: %s" % str(profile.get("questions_per_meeting", 0)),
			"Fillers per 100 words: %s" % str(profile.get("fillers_per_100_words", 0)),
		]
		if profile.get("top_mood") != null:
			lines.append("Usual mood: %s" % str(profile.top_mood))
		var pt: Array = profile.get("top_topics", [])
		if not pt.is_empty():
			lines.append("Usual topics: " + ", ".join(pt))
		if profile.get("last_seen") != null:
			lines.append("Last seen: " + Palette.date(float(profile.last_seen)))
		_profile.text = "\n".join(lines)
