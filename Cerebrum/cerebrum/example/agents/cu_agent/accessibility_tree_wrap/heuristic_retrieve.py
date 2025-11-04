import io
import xml.etree.ElementTree as ET
from typing import Tuple, List, Dict, Any
import copy # Ensure copy is imported at the top level

from PIL import Image, ImageDraw, ImageFont
import base64

import tiktoken

# --- Your existing namespaces ---
state_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/state"
state_ns_windows = "https://accessibility.windows.example.org/ns/state"
component_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/component"
component_ns_windows = "https://accessibility.windows.example.org/ns/component"
value_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/value"
value_ns_windows = "https://accessibility.windows.example.org/ns/value"
class_ns_windows = "https://accessibility.windows.example.org/ns/class"
attributes_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/attributes"
attributes_ns_windows = "https://accessibility.windows.example.org/ns/attributes"


# --- Modified judge_node function ---
def judge_node(node: ET.Element, platform="ubuntu", check_image=False) -> bool:
    if platform == "ubuntu":
        _state_ns = state_ns_ubuntu
        _component_ns = component_ns_ubuntu
    elif platform == "windows":
        _state_ns = state_ns_windows
        _component_ns = component_ns_windows
    else:
        raise ValueError("Invalid platform, must be 'ubuntu' or 'windows'")

    interactive_tags_suffixes = [
        "button", "item", "tabelement", "textfield", "textarea", "searchbox", "textbox", "link",
        "scrollbar",
    ]
    interactive_exact_tags = {
        "alert", "canvas", "check-box", "combo-box", "entry", "icon", "image",
        "menu", "menu bar", "popup menu",
        "radio button", "spin button",
        "slider", "static", "table-cell", "terminal", "text",
        "tree item", "tree table cell",
        "netuiribbontab", "start", "trayclockwclass", "traydummysearchcontrol",
        "uiimage", "uiproperty", "uiribboncommandbar", "document", "heading", "label", "paragraph",
        "scroll-bar", "section"
    }

    node_tag_lower = node.tag.lower()
    is_interactive_tag = any(node_tag_lower.endswith(suffix) for suffix in interactive_tags_suffixes) or \
                         node_tag_lower in interactive_exact_tags

    if not is_interactive_tag:
        return False

    keeps_state = False
    if platform == "ubuntu":
        keeps_state = (node.get(f"{{{_state_ns}}}showing", "false") == "true" and \
                       node.get(f"{{{_state_ns}}}visible", "false") == "true")
    elif platform == "windows":
        keeps_state = node.get(f"{{{_state_ns}}}visible", "false") == "true"

    if not keeps_state:
        return False

    is_enabled_or_interactive = (
        node.get(f"{{{_state_ns}}}enabled", "false") == "true" or
        node.get(f"{{{_state_ns}}}editable", "false") == "true" or
        node.get(f"{{{_state_ns}}}expandable", "false") == "true" or
        node.get(f"{{{_state_ns}}}checkable", "false") == "true" or
        node.get(f"{{{_state_ns}}}focusable", "false") == "true" or
        node.get(f"{{{_state_ns}}}selectable", "false") == "true" or
        node.get(f"{{{_state_ns}}}clickable", "false") == "true"
    )
    if not is_enabled_or_interactive:
        return False # Explicitly return False if no interactive state is true

    has_identifier = (
        node.get("name", "") != "" or
        (node.text is not None and len(node.text.strip()) > 0) or
        (check_image and node.get("image", "false") == "true")
    )
    if not has_identifier:
        return False # If not check_image, and no name/text, it's not identifiable for interaction

    coordinates_str = node.get(f"{{{_component_ns}}}screencoord", "(-1, -1)")
    sizes_str = node.get(f"{{{_component_ns}}}size", "(-1, -1)")

    try:
        coordinates: Tuple[int, int] = eval(coordinates_str)
        sizes: Tuple[int, int] = eval(sizes_str)
    except:
        return False

    if not (coordinates[0] >= 0 and coordinates[1] >= 0 and sizes[0] > 0 and sizes[1] > 0):
        return False

    return True

# --- New function to get interactive leaf elements for Ubuntu (returns dicts) ---
def get_ubuntu_interactive_leaf_elements(xml_file_str: str) -> List[Dict[str, Any]]:
    if not xml_file_str:
        return []

    root = ET.fromstring(xml_file_str)
    interactive_leaf_elements: List[Dict[str, Any]] = []

    for node in root.iter():
        # For this specific function's purpose, check_image is typically False,
        # as it's about the semantic tree structure, not visual identification via image.
        if judge_node(node, platform="ubuntu", check_image=False):
            has_interactive_child = False
            for child in node:
                if judge_node(child, platform="ubuntu", check_image=False):
                    has_interactive_child = True
                    break
            if not has_interactive_child:
                node_type = node.tag
                description = node.get("name", "").strip()
                if not description and node.text:
                    description = node.text.strip()

                coords_str = node.get(f"{{{component_ns_ubuntu}}}screencoord")
                size_str = node.get(f"{{{component_ns_ubuntu}}}size")

                if coords_str and size_str:
                    try:
                        coords = eval(coords_str)
                        size = eval(size_str)
                        location = (coords[0], coords[1], size[0], size[1])
                        interactive_leaf_elements.append({
                            "type": node_type,
                            "description": description or "",
                            "location": location
                        })
                    except Exception:
                        pass
    return interactive_leaf_elements


def find_leaf_nodes(xlm_file_str): # Original function, kept for other uses
    if not xlm_file_str:
        return []
    root = ET.fromstring(xlm_file_str)
    def collect_leaf_nodes(node, leaf_nodes):
        if not list(node): # Checks for XML children, not interactive children
            leaf_nodes.append(node)
        for child in node:
            collect_leaf_nodes(child, leaf_nodes)
    leaf_nodes = []
    collect_leaf_nodes(root, leaf_nodes)
    return leaf_nodes

# filter_nodes is used by linearize_accessibility_tree, keep it.
# For tag_screenshot, we'll use a more specific filtering for interactive leaf nodes.
def filter_nodes(root: ET.Element, platform="ubuntu", check_image=False):
    filtered_nodes = []
    for node in root.iter():
        if judge_node(node, platform, check_image):
            filtered_nodes.append(node)
    return filtered_nodes

def draw_bounding_boxes(nodes: List[ET.Element], image_file_content, down_sampling_ratio=1.0, platform="ubuntu"):
    if platform == "ubuntu":
        _state_ns = state_ns_ubuntu
        _component_ns = component_ns_ubuntu
        _value_ns = value_ns_ubuntu
        # _class_ns_to_use = attributes_ns_ubuntu # Not directly used in this simplified version
    elif platform == "windows":
        _state_ns = state_ns_windows
        _component_ns = component_ns_windows
        _value_ns = value_ns_windows
        # _class_ns_to_use = class_ns_windows # Not directly used
    else:
        raise ValueError("Invalid platform, must be 'ubuntu' or 'windows'")

    image_stream = io.BytesIO(image_file_content)
    image = Image.open(image_stream)
    if float(down_sampling_ratio) != 1.0:
        image = image.resize((int(image.size[0] * down_sampling_ratio), int(image.size[1] * down_sampling_ratio)))
    draw = ImageDraw.Draw(image)
    marks = []
    drew_nodes = []
    text_informations: List[str] = ["index\ttag\tname\ttext"]

    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    index = 1
    for _node in nodes:
        coords_str = _node.attrib.get(f'{{{_component_ns}}}screencoord')
        size_str = _node.attrib.get(f'{{{_component_ns}}}size')

        if coords_str and size_str:
            try:
                coords_tuple = eval(coords_str) # Keep original variable name distinct
                size_tuple = eval(size_str)     # Keep original variable name distinct

                original_coords = copy.deepcopy(coords_tuple)
                original_size = copy.deepcopy(size_tuple)

                # Apply downsampling
                current_coords = tuple(int(coord * down_sampling_ratio) for coord in coords_tuple)
                current_size = tuple(int(s * down_sampling_ratio) for s in size_tuple)


                if current_size[0] <= 0 or current_size[1] <= 0: continue

                bottom_right = (current_coords[0] + current_size[0], current_coords[1] + current_size[1])
                if bottom_right[0] < current_coords[0] or bottom_right[1] < current_coords[1]: continue

                cropped_image = image.crop((*current_coords, *bottom_right))
                if cropped_image.width > 0 and cropped_image.height > 0: # Ensure crop area is valid
                    extrema = cropped_image.getextrema()
                    is_truly_single_color = True
                    if len(extrema) > 0: # Check if getextrema returned valid data
                        for i in range(min(len(extrema),3)): # Check R, G, B bands primarily
                            if extrema[i][0] != extrema[i][1]:
                                is_truly_single_color = False
                                break
                        if is_truly_single_color and (current_size[0] * current_size[1] < 10000): # Avoid removing large single-color areas
                             continue
                    elif len(set(list(cropped_image.getdata()))) == 1 and (current_size[0] * current_size[1] < 10000): # Fallback for very small images or single band
                        continue


                draw.rectangle([current_coords, bottom_right], outline="red", width=1)
                text_position = (current_coords[0], bottom_right[1])
                text_bbox: Tuple[int, int, int, int] = draw.textbbox(text_position, str(index), font=font, anchor="lb")
                draw.rectangle(text_bbox, fill='black')
                draw.text(text_position, str(index), font=font, anchor="lb", fill="white")

                marks.append([original_coords[0], original_coords[1], original_size[0], original_size[1]])
                drew_nodes.append(_node)

                node_text_content = ""
                if _node.text and _node.text.strip():
                    node_text_content = _node.text.strip()
                elif platform == "windows" and _node.get(f"{{{class_ns_windows}}}", "").endswith("EditWrapper") \
                        and _node.get(f"{{{_value_ns}}}value"): # Use class_ns_windows for this Windows-specific check
                    node_text_content = _node.get(f"{{{_value_ns}}}value", "")
                
                name_attr_val = _node.get("name", "") # Use a distinct variable name

                # Quote fields if they contain quotes, or if they are the designated 'text' or 'name' field for consistency
                # if '"' in node_text_content:
                #     node_text_content_formatted = f"""
                #     "{node_text_content.replace("\"", "\"\"")}"
                #     """
                # else:
                #     # Optionally always quote text, or quote if it's not empty, or leave as is
                #     # For this case, if it's not empty or doesn't contain quotes, it's used as is.
                #     # The 'or '""'" later handles the empty case.
                #     node_text_content_formatted = node_text_content
                node_text_content_formatted = node_text_content

                # if '"' in name_attr_val:
                #      name_attr_formatted = f"""
                #      "{name_attr_val.replace("\"", "\"\"")}"
                #      """
                # else:
                #      name_attr_formatted = name_attr_val
                name_attr_formatted = name_attr_val
                
                # *** SYNTAX ERROR FIX IS HERE ***
                # Use '""' (a string of two double quotes) if node_text_content_formatted is empty.
                text_information: str = f"{index}\t{_node.tag}\t{name_attr_formatted}\t{node_text_content_formatted}"
                text_informations.append(text_information)
                index += 1
            except ValueError:
                pass
            except Exception:
                pass

    output_image_stream = io.BytesIO()
    image.save(output_image_stream, format='PNG')
    image_content = output_image_stream.getvalue()
    return marks, drew_nodes, "\n".join(text_informations), image_content

def print_nodes_with_indent(nodes: List[ET.Element], indent=0): # Added type hint for nodes
    for node in nodes:
        print(' ' * indent, node.tag, node.attrib)
        print_nodes_with_indent(list(node), indent + 2)

def encode_image(image_content):
    return base64.b64encode(image_content).decode('utf-8')

def linearize_accessibility_tree(accessibility_tree_str, platform="ubuntu"):
    if not accessibility_tree_str:
        return "tag\tname\ttext\tclass\tdescription\tposition (top-left x&y)\tsize (w&h)"

    root = ET.fromstring(accessibility_tree_str)
    if platform == "ubuntu":
        _attributes_ns = attributes_ns_ubuntu
        # _state_ns = state_ns_ubuntu # Not used directly here
        _component_ns = component_ns_ubuntu
        # _value_ns = value_ns_ubuntu # Not used directly here
        _class_ns_to_use = _attributes_ns
    elif platform == "windows":
        _attributes_ns = attributes_ns_windows
        # _state_ns = state_ns_windows # Not used
        _component_ns = component_ns_windows
        _value_ns = value_ns_windows # Used for windows text fallback
        _class_ns_to_use = class_ns_windows
    else:
        raise ValueError("Invalid platform, must be 'ubuntu' or 'windows'")

    # filter_nodes uses check_image=False by default, which is appropriate for semantic linearization
    filtered_nodes = filter_nodes(root, platform, check_image=False)
    linearized_list = ["tag\tname\ttext\tclass\tdescription\tposition (top-left x&y)\tsize (w&h)"]

    for node in filtered_nodes:
        text_content = ""
        if node.text and node.text.strip():
            text_content = node.text.strip()
        elif platform == "windows" and node.get(f"{{{class_ns_windows}}}", "").endswith("EditWrapper") \
                and node.get(f"{{{_value_ns}}}value"): # Use class_ns_windows here
            text_content = node.get(f"{{{_value_ns}}}value", "")
        
        name_val = node.get("name", "")
        text_content_formatted = text_content # placeholder for consistent naming if more formatting added
        name_val_formatted = name_val       # placeholder

        # Basic CSV-like quoting for fields that might contain the delimiter or quotes
        # if '"' in text_content_formatted or '\t' in text_content_formatted:
        #      text_content_formatted = f'"{text_content_formatted.replace("\"", "\"\"")}"'
        # elif not text_content_formatted: # Represent empty text as ""
        #     text_content_formatted = '""'

        text_content_formatted = text_content
        # if '"' in name_val_formatted or '\t' in name_val_formatted:
        #     name_val_formatted = f'"{name_val_formatted.replace("\"", "\"\"")}"'
        # elif not name_val_formatted: # Represent empty name as ""
        #     name_val_formatted = '""'
        name_val_formatted = name_val
        
        
        class_attr = node.get(f"{{{_class_ns_to_use}}}class", "")
        if platform == "ubuntu" and not class_attr:
             class_attr = node.get("class", "") # Non-namespaced fallback

        description_attr = node.get(f"{{{_attributes_ns}}}description", "")
        if not description_attr:
            description_attr = node.get("description","") # Non-namespaced fallback
        if not class_attr: class_attr = '""'
        if not description_attr: description_attr = '""'


        linearized_list.append(
            f"{node.tag}\t{name_val_formatted}\t{text_content_formatted}\t"
            f"{class_attr}\t{description_attr}\t"
            f"{node.get(f'{{{_component_ns}}}screencoord', '(-1,-1)')}\t" # provide default
            f"{node.get(f'{{{_component_ns}}}size', '(-1,-1)')}"  # provide default
        )
    return "\n".join(linearized_list)

def trim_accessibility_tree(linearized_accessibility_tree, max_tokens):
    # Assuming tiktoken is correctly installed and gpt-4 model is appropriate
    try:
        enc = tiktoken.encoding_for_model("gpt-4")
    except Exception: # Fallback if tiktoken or model name is problematic
        enc = tiktoken.get_encoding("cl100k_base") # A common encoding

    tokens = enc.encode(linearized_accessibility_tree)
    if len(tokens) > max_tokens:
        # Ensure decoding doesn't cut mid-character if possible, though tiktoken handles this well.
        trimmed_text = enc.decode(tokens[:max_tokens])
        # Find the last newline to avoid cutting a line mid-way.
        last_newline = trimmed_text.rfind('\n')
        if last_newline != -1:
            trimmed_text = trimmed_text[:last_newline]
        linearized_accessibility_tree = trimmed_text + "\n[...]\n"
    return linearized_accessibility_tree

# --- MODIFIED tag_screenshot function ---
def tag_screenshot(screenshot, accessibility_tree_str, platform="ubuntu"):
    if not accessibility_tree_str:
        # Handle empty accessibility tree string
        tagged_screenshot_content = b""
        if screenshot:
            try:
                # Just return the original screenshot if no tree to process
                img_byte_arr = io.BytesIO()
                Image.open(io.BytesIO(screenshot)).save(img_byte_arr, format='PNG')
                tagged_screenshot_content = img_byte_arr.getvalue()
            except Exception: # If screenshot data is invalid
                 pass # tagged_screenshot_content remains b""
        return [], [], tagged_screenshot_content, "index\ttag\tname\ttext"

    interactive_leaf_nodes_for_drawing: List[ET.Element] = []
    try:
        root = ET.fromstring(accessibility_tree_str)
        for node_to_check in root.iter():
            # For drawing, check_image=True is important as elements might be identified by image
            if judge_node(node_to_check, platform=platform, check_image=True):
                is_leaf_for_drawing = True
                for child_node in node_to_check: # Iterate direct children
                    if judge_node(child_node, platform=platform, check_image=True):
                        is_leaf_for_drawing = False
                        break
                if is_leaf_for_drawing:
                    interactive_leaf_nodes_for_drawing.append(node_to_check)
    except ET.ParseError:
        # If tree is unparsable, behave like an empty tree for drawing
        tagged_screenshot_content = b""
        if screenshot:
            try:
                img_byte_arr = io.BytesIO()
                Image.open(io.BytesIO(screenshot)).save(img_byte_arr, format='PNG')
                tagged_screenshot_content = img_byte_arr.getvalue()
            except Exception:
                pass
        return [], [], tagged_screenshot_content, "index\ttag\tname\ttext"

    # Proceed to draw bounding boxes on the found interactive leaf nodes
    marks, drew_nodes, element_list, tagged_screenshot_content = draw_bounding_boxes(
        interactive_leaf_nodes_for_drawing, screenshot, platform=platform
    )
    return marks, drew_nodes, tagged_screenshot_content, element_list

# --- Example Usage (assuming you have an XML string) ---
if __name__ == '__main__':
    ns_map = {
        'comp': component_ns_ubuntu,
        'state': state_ns_ubuntu,
        'attr': attributes_ns_ubuntu
    }
    mock_ubuntu_xml_str = f"""<?xml version="1.0" encoding="UTF-8"?>
<application name="Test App">
  <frame name="Main Window" {{{ns_map['state']}:visible="true" {{{ns_map['state']}:showing="true" {{{ns_map['comp']}:screencoord="(0,0)" {{{ns_map['comp']}:size="(800,600)">
    <panel {{{ns_map['state']}:visible="true" {{{ns_map['state']}:showing="true" {{{ns_map['comp']}:screencoord="(10,10)" {{{ns_map['comp']}:size="(780,580)">
      <button name="Click Me" {{{ns_map['state']}:visible="true" {{{ns_map['state']}:showing="true" {{{ns_map['state']}:enabled="true" {{{ns_map['comp']}:screencoord="(50,50)" {{{ns_map['comp']}:size="(100,30)">
        <label {{{ns_map['state']}:visible="true" {{{ns_map['state']}:showing="true" {{{ns_map['comp']}:screencoord="(55,55)" {{{ns_map['comp']}:size="(90,20)">Click Me Text</label>
      </button>
      <text name="Username" {{{ns_map['state']}:visible="true" {{{ns_map['state']}:showing="true" {{{ns_map['state']}:enabled="true" {{{ns_map['state']}:editable="true" {{{ns_map['comp']}:screencoord="(50,100)" {{{ns_map['comp']}:size="(150,25)" />
      <check-box name="Agree to terms" {{{ns_map['state']}:visible="true" {{{ns_map['state']}:showing="true" {{{ns_map['state']}:enabled="true" {{{ns_map['state']}:checkable="true" {{{ns_map['comp']}:screencoord="(50,150)" {{{ns_map['comp']}:size="(180,25)" />
      <panel name="InteractiveParentContainer" {{{ns_map['state']}:visible="true" {{{ns_map['state']}:showing="true" {{{ns_map['state']}:enabled="true" {{{ns_map['state']}:focusable="true" {{{ns_map['comp']}:screencoord="(50,200)" {{{ns_map['comp']}:size="(200,100)">
         <button name="Nested Button" {{{ns_map['state']}:visible="true" {{{ns_map['state']}:showing="true" {{{ns_map['state']}:enabled="true" {{{ns_map['comp']}:screencoord="(60,210)" {{{ns_map['comp']}:size="(100,30)" />
      </panel>
      <button name="Disabled Button" {{{ns_map['state']}:visible="true" {{{ns_map['state']}:showing="true" {{{ns_map['state']}:enabled="false" {{{ns_map['comp']}:screencoord="(50,350)" {{{ns_map['comp']}:size="(100,30)" />
      <button name="" {{{ns_map['state']}:visible="true" {{{ns_map['state']}:showing="true" {{{ns_map['state']}:enabled="true" {{{ns_map['comp']}:screencoord="(50,400)" {{{ns_map['comp']}:size="(100,30)">Implicit Name from Text</button>
    </panel>
  </frame>
</application>
    """.replace("{{ ","{").replace(" }}","}")

    print("---- Mock Ubuntu XML ----")
    
    print("\n---- judge_node tests ----")
    try:
        root_for_test = ET.fromstring(mock_ubuntu_xml_str)
        first_button = root_for_test.find(".//*[@name='Click Me']") # More robust XPath
        if first_button is not None:
            print(f"Is 'Click Me' button interactive? {judge_node(first_button, platform='ubuntu', check_image=True)}")
        
        text_input = root_for_test.find(".//*[@name='Username']")
        if text_input is not None:
            print(f"Is 'Username' text input interactive? {judge_node(text_input, platform='ubuntu')}")

        disabled_button = root_for_test.find(".//*[@name='Disabled Button']")
        if disabled_button is not None:
            print(f"Is 'Disabled Button' interactive? {judge_node(disabled_button, platform='ubuntu')}")

        panel_with_button = root_for_test.find(".//*[@name='InteractiveParentContainer']")
        if panel_with_button is not None:
             print(f"Is 'InteractiveParentContainer' panel interactive? {judge_node(panel_with_button, platform='ubuntu', check_image=True)}") # Should be true
             # Test if it's a leaf for drawing
             is_leaf_for_drawing = True
             for child_node in panel_with_button:
                 if judge_node(child_node, platform='ubuntu', check_image=True):
                     is_leaf_for_drawing = False; break
             print(f"Is 'InteractiveParentContainer' panel an interactive LEAF for drawing? {is_leaf_for_drawing}") # Should be False

    except ET.ParseError as e:
        print(f"Error parsing mock XML for judge_node test: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during judge_node tests: {e}")

    print("\n---- Interactive Leaf Elements for Ubuntu (List of Dictionaries) ----")
    try:
        interactive_elements_data = get_ubuntu_interactive_leaf_elements(mock_ubuntu_xml_str)
        if interactive_elements_data:
            for elem in interactive_elements_data:
                print(elem)
        else:
            print("No interactive leaf elements found or XML was empty/invalid.")
    except ET.ParseError as e:
        print(f"Error parsing mock XML for get_ubuntu_interactive_leaf_elements: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print("\n---- Linearized Accessibility Tree for Ubuntu ----")
    try:
        linearized_tree = linearize_accessibility_tree(mock_ubuntu_xml_str, platform="ubuntu")
        print(linearized_tree)
    except ET.ParseError as e:
        print(f"Error parsing mock XML for linearization: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during linearization: {e}")

    # Example of using tag_screenshot (requires a mock image)
    print("\n---- Tag Screenshot (with interactive leaf elements) ----")
    try:
        # Create a dummy 1x1 black PNG image for testing
        dummy_image = Image.new('RGB', (100, 100), (0, 0, 0))
        img_byte_arr = io.BytesIO()
        dummy_image.save(img_byte_arr, format='PNG')
        dummy_image_content = img_byte_arr.getvalue()

        marks, drew_nodes, tagged_img_content, element_list_str = tag_screenshot(
            dummy_image_content, mock_ubuntu_xml_str, platform="ubuntu"
        )
        print(f"Number of marks made: {len(marks)}")
        print("Element list from tagging:")
        print(element_list_str)
        # To save/view tagged_img_content:
        # with open("tagged_screenshot.png", "wb") as f:
        #     f.write(tagged_img_content)
        # print("Saved tagged_screenshot.png")

    except ET.ParseError as e:
        print(f"Error parsing mock XML for tag_screenshot: {e}")
    except ImportError:
        print("Pillow (PIL) or other imaging library might be missing for tag_screenshot.")
    except Exception as e:
        print(f"An unexpected error occurred during tag_screenshot: {e}")