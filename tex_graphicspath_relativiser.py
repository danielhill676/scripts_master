import os
import re
import shutil

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

tex_file = "/Users/administrator/Astro/github/Paper_1/mnras_template.tex"

old_base = "/Users/administrator/Astro/LLAMA/ALMA/"
paper_dir = "/Users/administrator/Astro/github/Paper_1/"

# ------------------------------------------------------------
# Read .tex file
# ------------------------------------------------------------

with open(tex_file, "r", encoding="utf-8") as f:
    tex = f.read()


# debug

# print(f"File length: {len(tex)} characters")
# print(f"Number of literal 'includegraphics' strings: {tex.count('includegraphics')}")
# print()

# # Show every occurrence, including its surrounding text
# for match in re.finditer(r"includegraphics", tex):
#     start = max(0, match.start() - 100)
#     end = min(len(tex), match.end() + 200)

#     print("------------------------------------------------------------")
#     print(repr(tex[start:end]))



# ------------------------------------------------------------
# Find all \includegraphics paths
# ------------------------------------------------------------

#pattern = r"(\\includegraphics(?:\[[^\]]*\])?\{)([^}]+)(\})"
pattern = r"(\\includegraphics\*?\s*(?:\[[^\]]*\])?\s*\{)([^}]+)(\})"

matches = list(re.finditer(pattern, tex))

print(f"Found {len(matches)} \\includegraphics calls.\n")

# Keep track of copied files
copied = []
missing = []
modified = 0

# ------------------------------------------------------------
# Replace paths
# ------------------------------------------------------------

def replace_graphics_path(match):

    global modified

    prefix = match.group(1)
    src = match.group(2)
    suffix = match.group(3)

    # --------------------------------------------------------
    # Normalise accidental double slash at beginning
    # --------------------------------------------------------

    normalised_src = src.replace("//Users/", "/Users/")

    # --------------------------------------------------------
    # Only modify paths originating from old_base
    # --------------------------------------------------------

    if not normalised_src.startswith(old_base):
        print(f"SKIPPING (not from expected base):")
        print(f"    {src}\n")
        return match.group(0)

    # --------------------------------------------------------
    # Get path relative to LLAMA/ALMA
    # --------------------------------------------------------

    relative_path = normalised_src[len(old_base):]

    # --------------------------------------------------------
    # Destination inside Paper_1
    # --------------------------------------------------------

    destination = os.path.join(
        paper_dir,
        relative_path
    )

    # --------------------------------------------------------
    # Create destination directory
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(destination),
        exist_ok=True
    )

    # --------------------------------------------------------
    # Copy the file
    # --------------------------------------------------------

    if os.path.isfile(normalised_src):

        shutil.copy2(
            normalised_src,
            destination
        )

        copied.append(relative_path)

        print("COPIED:")
        print(f"    {normalised_src}")
        print(f" -> {destination}")

    else:

        missing.append(normalised_src)

        print("MISSING:")
        print(f"    {normalised_src}")

    # --------------------------------------------------------
    # Return the relative path for the .tex file
    # --------------------------------------------------------

    modified += 1

    return prefix + relative_path + suffix


# ------------------------------------------------------------
# Perform replacements
# ------------------------------------------------------------

new_tex = re.sub(
    pattern,
    replace_graphics_path,
    tex
)

# ------------------------------------------------------------
# Write modified .tex file
# ------------------------------------------------------------

with open(tex_file, "w", encoding="utf-8") as f:
    f.write(new_tex)

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("Finished")
print("=" * 60)

print(f"\nincludegraphics calls found : {len(matches)}")
print(f"Paths modified              : {modified}")
print(f"Files copied                : {len(copied)}")
print(f"Files missing               : {len(missing)}")

if missing:
    print("\nMissing files:")
    for filename in missing:
        print(f"    {filename}")