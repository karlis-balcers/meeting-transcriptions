class_name Avatars
extends RefCounted
## Profile pictures, one per speaker name, kept next to the app's own data.
## Pictures are cropped to a circle when they're added, so the stage can draw
## them straight into the bubble.

const DIR := "user://avatars"
const SIZE := 256

static var _cache := {}  # name -> Texture2D, or null when there is none


static func _path(name: String) -> String:
	return "%s/%s.png" % [DIR, name.strip_edges().to_lower().md5_text()]


static func texture(name: String) -> Texture2D:
	if _cache.has(name):
		return _cache[name]
	var tex: Texture2D = null
	var path := _path(name)
	if FileAccess.file_exists(path):
		var img := Image.load_from_file(path)
		if img != null and not img.is_empty():
			tex = ImageTexture.create_from_image(img)
	_cache[name] = tex
	return tex


static func has(name: String) -> bool:
	return texture(name) != null


## Crop the picture at `file` to a centered circle and store it for `name`.
## Returns an error message, or "" when it worked.
static func set_from_file(name: String, file: String) -> String:
	var img := Image.load_from_file(file)
	if img == null or img.is_empty():
		return "Couldn't read that picture. PNG, JPG and WebP work."
	var side := mini(img.get_width(), img.get_height())
	img = img.get_region(Rect2i((img.get_width() - side) / 2, (img.get_height() - side) / 2, side, side))
	img.convert(Image.FORMAT_RGBA8)
	img.resize(SIZE, SIZE, Image.INTERPOLATE_LANCZOS)
	var c := (SIZE - 1) / 2.0
	for y in SIZE:
		for x in SIZE:
			var d := Vector2(x - c, y - c).length()
			var a := clampf(c - d + 0.5, 0.0, 1.0)  # soft 1px edge
			if a < 1.0:
				var px := img.get_pixel(x, y)
				px.a *= a
				img.set_pixel(x, y, px)
	DirAccess.make_dir_recursive_absolute(DIR)
	var err := img.save_png(_path(name))
	if err != OK:
		return "Couldn't save the picture (error %d)." % err
	_cache[name] = ImageTexture.create_from_image(img)
	return ""


static func remove(name: String) -> void:
	DirAccess.remove_absolute(ProjectSettings.globalize_path(_path(name)))
	_cache[name] = null


## Follow a rename. When the new name already has a picture, it keeps its own.
static func rename(old: String, new_name: String) -> void:
	var src := _path(old)
	if old == new_name or not FileAccess.file_exists(src):
		return
	if not FileAccess.file_exists(_path(new_name)):
		DirAccess.rename_absolute(ProjectSettings.globalize_path(src), ProjectSettings.globalize_path(_path(new_name)))
	else:
		DirAccess.remove_absolute(ProjectSettings.globalize_path(src))
	_cache.erase(old)
	_cache.erase(new_name)
