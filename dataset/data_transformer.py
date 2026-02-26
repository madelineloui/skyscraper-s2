## Will transfer the data from the csv/xlsx file into a json file that
## is formatted in the way that the teochat_intruct file outlines so that the
## teochat LLM can read it

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import re
import json
import random


def _to_iso_date(x: Any) -> Optional[str]:
    """
    Convert a CSV cell into 'YYYY-MM-DD' if possible, else None.
    Works for strings like '2022-09-05' or Excel-like values.
    """
    if pd.isna(x) or x == "":
        return None
    ts = pd.to_datetime(x, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.date().isoformat()


def load_labels_as_dicts(labels_csv_path: str | Path) -> List[Dict[str, Any]]:
    """
    Reads labels.csv and returns a list of dictionaries (one per row).
    Keys are normalized to snake_case, values come from that row.
    Also converts *date columns* to ISO strings if they exist.
    """
    labels_csv_path = Path(labels_csv_path)
    df = pd.read_csv(labels_csv_path)

    # Normalize column names: strip + lower + spaces -> underscores
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Make sure article_id is always a string (helps later when matching folder names)
    if "article_id" in df.columns:
        df["article_id"] = df["article_id"].astype(str)

    # If your CSV has these date columns (names based on your screenshot),
    # convert them to ISO format. If some don’t exist, it's fine.
    possible_date_cols = [
        "event_start_date",
        "event_end_date",
        "sequence_start_date",
        "sequence_end_date",
        # If your CSV uses slightly different names, add them here.
        # e.g. "event_start", "event_end", "sequence_start", "sequence_end"
    ]
    for col in possible_date_cols:
        if col in df.columns:
            df[col] = df[col].apply(_to_iso_date)

    # Convert to list-of-dicts
    records: List[Dict[str, Any]] = df.to_dict(orient="records")
    return records

def get_lat_long(entry):
    '''
    Takes in a dictionary entry for the latitude and longitude for an
    article_id in the form "lat_long" and returns two floats in a tuple
    representing the longitude and latitude

    Returns: (latitude, longitude) --> tuple of floats
    '''

    index = entry.find('_')
    latitude = float(entry[:index])
    longitude = float(entry[index + 1:])
    return latitude, longitude

def revised_dictionary(labels_path):
    '''
    Takes the dictionary created by the function load_labels_as_dicts
    and creates a new list of dictionaries that is slightly better suited for the
    specific json format

    Returns: revised list of dictionaries --> list
    '''

    loaded_list = load_labels_as_dicts(labels_path)
    revised_list = []

    for d in loaded_list:
        current_dict = {}
        current_dict['article_id'] = d['article_id']
        current_dict['event_type'] = d['event_type']
        current_dict['event_caption'] = d['event_caption']
        current_dict['visible'] = d['visible']
        current_dict['sequence_start_date'] = d['sequence_start_date']
        current_dict['sequence_end_date'] = d['sequence_end_date']
        current_dict['event_start_date'] = d['event_start_date']
        current_dict['event_end_date'] = d['event_end_date']
        current_dict['location_name'] = d['location_name']
        lat, long = get_lat_long(d['coordinates'])
        current_dict['latitude'] = lat
        current_dict['longitude'] = long
        current_dict['source'] = d['source']
        current_dict['api'] = d['api']
        current_dict['url'] = d['url']
        current_dict['article_content'] = d['article_content']
        revised_list.append(current_dict)

    return revised_list

DATE_RE = re.compile(r"^-?\d+(?:\.\d+)?_-?\d+(?:\.\d+)?_(\d{8})(?:_|\.|$)")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def yyyymmdd_to_iso(yyyymmdd: str) -> str:
    """Convert 'YYYYMMDD' -> 'YYYY-MM-DD'."""
    return f"{yyyymmdd[0:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"


def extract_yyyymmdd_from_filename(filename: str) -> Optional[str]:
    """
    Extract the YYYYMMDD portion from filenames like:
      40.1398_-120.9509_20210708_190325.jpg
    Returns 'YYYYMMDD' or None if it doesn't match.
    """
    m = DATE_RE.match(filename)
    if not m:
        return None
    return m.group(1)


def list_image_files_for_article(imagery_dir, article_id: str) -> List[Path]:
    """
    Returns a list of image file Paths under imagery_dir/article_id/.
    Filters by common image extensions.
    """
    imagery_dir = Path(imagery_dir)
    folder = imagery_dir / str(article_id)

    if not folder.exists() or not folder.is_dir():
        return []

    files = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    return files

## Gets the number of images for the specific article_id and
## gets the dates of that image in a list
def get_image_dates_for_article(
    imagery_dir,
    article_id: str,
    *,
    sort: bool = True,
    unique: bool = False,
):
    """
    Fetch images for an article_id and return:
      (num_images, dates_iso_list)

    dates_iso_list is derived from the YYYYMMDD segment after the second underscore.
    Example output: (8, ['2021-07-01', '2021-07-08', ...])

    Params:
      - sort: sort chronologically (recommended)
      - unique: if True, de-duplicate dates (keeps first occurrence)
    """
    image_files = list_image_files_for_article(imagery_dir, article_id)

    dates: List[str] = []
    for p in image_files:
        yyyymmdd = extract_yyyymmdd_from_filename(p.name)
        if yyyymmdd is None:
            # Skip unexpected filenames
            continue
        dates.append(yyyymmdd_to_iso(yyyymmdd))

    if sort:
        dates.sort()

    if unique:
        seen = set()
        deduped = []
        for d in dates:
            if d not in seen:
                seen.add(d)
                deduped.append(d)
        dates = deduped

    return len(image_files), dates

def get_event_types(labels_path):
    '''
    Gets all the possible event types to be used for event classification

    Returns: set of all possible event_types --> set
    '''
    all_articles = revised_dictionary(labels_path)

    all_types = set()

    for article in all_articles:
        all_types.add(article['event_type'])

    return all_types

ALL_EVENT_TYPES = {'beach nourishment', 'water level decrease', 'storm', 'vessel withdrawal', 'truck congestion', 'defoliation', 'maritime activity reduction', 'land exposure', 'Maritime gathering', 'boat activity', 'site preparation', 'earthquake', 'Dam destruction', 'Airshow', 'recovery operation', 'harvesting', 'infrastructure presence', 'bridge completion', 'temporary structure', 'excavation', 'Bridge replacement', 'expansion', 'military deployment', 'burn scar', 'bridge damage', 'temporary settlement', 'bunkering', 'wildfire aftermath', 'vehicle presence', 'camp establishment', 'winter weather', 'testing', 'coastal structure', 'logistics', 'cyclone damage', 'Mudslide', 'dry conditions', 'resettlement', 'Vessel departure', 'Convention', 'Inundation', 'highway collapse', 'Erosion', 'vessel activity reduction', 'mining activity', 'snow melt', 'diversion', 'drydock', 'public event', 'Sediment plume', 'traffic reduction', 'coastal changes', 'tree removal', 'grounds damage', 'Infrastructure damage', 'vessel mooring', 'Ground disturbance', 'maritime activity', 'deforestation', 'shipping', 'pond change', 'Water discoloration', 'mining', 'reconstruction', 'Turbidity', 'contamination', 'burn mark', 'launch preparation', 'maritime incident', 'assembly preparation', 'land clearing', 'Installation', 'road closure', 'road transformation', 'Water disturbance', 'field alteration', 'sea conditions', 'displacement', 'reservoir filling', 'eclipse', 'fairground', 'vessel destruction', 'earthquake damage', 'Displacement', 'aircraft departure', 'environmental impact', 'traffic congestion', 'structure building', 'fire break', 'Harvest', 'Naval visit', 'barrier construction', 'Smoke', 'Shipbuilding', 'reservoir draining', 'haze', 'clearing', 'loading', 'naval presence', 'inactivity', 'derailment aftermath', 'vessel return', 'vessel gathering', 'Shipping activity', 'emergency response', 'fire', 'shipping congestion', 'Exhibition', 'Bridge collapse', 'Structural collapse', 'coastal alteration', 'Aircraft relocation', 'burned area', 'building destruction', 'sailing event', 'water color change', 'derailment', 'Activity', 'Temporary structures', 'coastal change', 'port activity reduction', 'road modification', 'traffic increase', 'Eradication', 'arson', 'deployment', 'conflict damage', 'naval incident', 'storm damage', 'Settlement expansion', 'land disturbance', 'Discharge', 'rocket launch', 'Commercial opening', 'congestion', 'Vessel', 'pond draining', 'Stalled development', 'Strike activity', 'Encampment development', 'Transshipment', 'pollution', 'installation', 'Event', 'military withdrawal', 'earthwork', 'demolition', 'seasonal change', 'Coastal change', 'Aftermath', 'tornado', 'waterlogging', 'water release', 'Logistics activity', 'structure destruction', 'coastal activity', 'military development', 'ceremonial setup', 'withdrawal', 'Impact', 'occupation', 'strike damage', 'site clearing', 'smoke plumes', 'settlement development', 'security deployment', 'Damage', 'wave activity', 'maritime gathering', 'Naval gathering', 'snow cover', 'rough surf', 'bridge destruction', 'ashfall', 'destruction', 'eruption', 'Aid delivery', 'Strike', 'ship visit', 'structural collapse', 'Building destruction', 'inauguration', 'volcanic eruption', 'agricultural change', 'decoration', 'storm impact', 'naval activity', 'truck queues', 'event preparation', 'landfill fire', 'fair', 'coastal protection', 'aircraft crash', 'River changes', 'industrial damage', 'Burning', 'smoke', 'aircraft increase', 'Sedimentation', 'expressway', 'Land clearing', 'Maritime event', 'flood damage', 'bombardment', 'ship departure', 'military crossing', 'ceremony', 'Conflict Damage', 'tornado damage', 'Bridge destruction', 'construction', 'Repaving', 'landing', 'naval vessel presence', 'burn scars', 'maritime transit', 'Vessel capsizing', 'coastal inundation', 'Public Gathering', 'vegetation stress', 'checkpoint establishment', 'Vessel congregation', 'crop change', 'water accumulation', 'River turbidity', 'Border activity', 'defense construction', 'Ground operations', 'increased water levels', 'Vessel launch', 'washout', 'rally', 'road paving', 'Event activity', 'coastal erosion', 'removal', 'smoke plume', 'Field treatment', 'dust storm', 'Timber storage', 'foliage change', 'Effluent discharge', 'Settlement', 'temporary construction', 'Naval vessel presence', 'fire suppression', 'Debris flow', 'airlift activity', 'rail transport', 'Riverbed exposure', 'public display', 'Commercial activity', 'field fire', 'Arson', 'Ship damage', 'Event infrastructure', 'Derailment', 'ground damage', 'high water', 'celebration', 'Traffic congestion', 'aircraft placement', 'Grain transport', 'vessel capsizing', 'Event setup', 'recreation', 'assembly', 'shelter construction', 'Fire damage', 'Shipping', 'land art', 'Water flow', 'water level increase', 'aircraft destruction', 'plume', 'Dredging', 'development', 'dam destruction', 'turbidity', 'ground clearing', 'Boating activity', 'Material dumping', 'Barrier installation', 'aid movement', 'impacts', 'border activity', 'component storage', 'tsunami damage', 'market activity', 'Smoke plumes', 'Crowd gathering', 'dam spill', 'festival setup', 'Launch', 'aerial activity', 'renourishment', 'crop damage', 'riverine construction', 'naval gathering', 'Snowfall', 'Boat activity', 'algal bloom', 'Festival', 'evacuation', 'coastal modification', 'cultivation', 'Investigation', 'pipeline', 'Storm impact', 'staging', 'strike', 'disturbance', 'Sediment discharge', 'Dam removal', 'Event Activity', 'Waterlogging', 'flood', 'stadium activity', 'crash site', 'shipyard activity', 'aircraft', 'inundation', 'Water release', 'cliff collapse', 'Recovery operation', 'flowering', 'dam removal', 'aircraft movement', 'vessel removal', 'cargo delivery', 'smoke haze', 'Urban development', 'Cargo transport', 'vehicle activity', 'Earthquake damage', 'ship presence', 'cruise activity', 'vessel damage', 'Water level drop', 'flood recession', 'incident', 'fog', 'vegetation damage', 'temporary installation', 'event infrastructure', 'landscaping', 'Snow accumulation', 'Shelter construction', 'dam breach', 'Fire destruction', 'material staging', 'Restoration', 'explosion damage', 'Arena activity', 'rail traffic', 'structural damage', 'vehicle reduction', 'debris', 'Vessel arrival', 'crash', 'ship movement', 'Display', 'Sailing', 'Urban destruction', 'Burial ground expansion', 'Smoke plume', 'shipping disruption', 'Industrial activity reduction', 'Dust storm', 'road blockage', 'docking', 'Parade preparation', 'Vessel activity', 'site alteration', 'Naval activity', 'aircraft deployment', 'water flow', 'launch aftermath', 'smog', 'airstrike', 'Construction', 'vessel docking', 'Burned area', 'crop loss', 'Road blockage', 'vehicle fire', 'bushfire', 'Encampment', 'Vessel anchoring', 'departure', 'wildflower bloom', 'explosion', 'bombing', 'Vessel detention', 'construction cessation', 'military activity', 'river alteration', 'Waterway maintenance', 'fire damage', 'Vessel docking', 'Urban Damage', 'Border traffic', 'Construction halt', 'roadwork', 'strike activity', 'Vehicle presence', 'burn', 'maritime departure', 'Security', 'Aircraft deployment', 'maritime exhibition', 'road construction', 'commemoration', 'structure erection', 'security measures', 'exhibition', 'Incendiary attack', 'Flood', 'aircraft presence', 'military exercise', 'carnival', 'Outdoor event', 'Vegetation clearing', 'Exploration', 'hurricane', 'market setup', 'frost', 'airport operation', 'blockade', 'Discoloration', 'debris flow', 'Flaring', 'site development', 'low water', 'Ship presence', 'atmospheric haze', 'public gathering', 'Flood damage', 'military presence', 'brush fire', 'Aircraft presence', 'winter storm', 'Vessel loading', 'market', 'water quality change', 'waste accumulation', 'canal breach', 'river level change', 'Water level increase', 'dam failure', 'burial', 'ship arrival', 'river changes', 'Ashfall', 'agricultural damage', 'Maritime activity', 'Water discharge', 'Vessel influx', 'sediment runoff', 'mass gathering', 'Track disturbance', 'Conflict', 'Salvage', 'aquaculture', 'vessel departure', 'River alteration', 'renovation', 'land development', 'gathering', 'infrastructure damage', 'harvest', 'river swelling', 'rail activity', 'greening', 'prescribed burn', 'Coastal flooding', 'attack aftermath', 'Repair', 'agricultural burning', 'traffic', 'snowmelt', 'mudflow', 'dismantling', 'crowd gathering', 'shipping activity', 'fire activity', 'drought', 'aircraft parking', 'water discharge', 'vessel', 'Development', 'Flooding and landslides', 'damage', 'Road construction', 'runoff', 'vehicle influx', 'storm surge', 'vegetation drying', 'Site preparation', 'canal repair', 'Volcanic eruption', 'burning', 'gathering preparation', 'Industrial smoke', 'Bridge construction', 'Shelling', 'wildfire smoke', 'barrier installation', 'Structural damage', 'Drought', 'Site Development', 'Vessel movement', 'structure removal', 'Industrial fire', 'infrastructure', 'Tornado', 'Structures', 'drying', 'building fire', 'drainage', 'vessel operation', 'vegetation change', 'Increased activity', 'Lahar flow', 'ground survey', 'recovery', 'coastal flooding', 'Flooding', 'port damage', 'attack', 'Docking', 'Port activity', 'Vehicle activity', 'Logistics', 'housing', 'Vessel delivery', 'vessel dispersal', 'structure fire', 'roofing', 'event setup', 'air activity', 'temporary structures', 'Environmental change', 'airport activity', 'impact damage', 'Strikes', 'encampment', 'mudslide', 'shipping incident', 'burial site expansion', 'convention', 'Gathering', 'Explosion', 'landslides', 'Structure removal', 'water recession', 'Fires', 'dredging', 'Snowmelt', 'Vehicle queues', 'activity', 'banner display', 'vessel relocation', 'Dam failure', 'spill', 'building collapse', 'Burn scar', 'preparation', 'planting', 'vessel activity', 'Low water', 'Military deployment', 'discharge', 'water level change', 'typhoon damage', 'vehicle removal', 'Smog', 'Eclipse', 'event activity', 'vessel congregation', 'snowfall', 'conflict', 'Blockade', 'oil spill', 'ferry operation', 'Coastal turbidity', 'Conflict damage', 'flooding', 'debris disposal', 'Riverbed change', 'ship grounding', 'reduced activity', 'Burn', 'temporary shelters', 'road painting', 'facility development', 'ship launch', 'Reconstruction', 'maritime traffic', 'Vessel presence', 'port activity', 'emergency facility', 'river level decrease', 'industrial activity', 'parade', 'vessel fire', 'Infrastructure', 'Vehicle increase', 'Vegetation health', 'Spill response', 'volcanic impact', 'vessel construction', 'Demolition', 'erosion', 'modification', 'Dam breach', 'discoloration', 'burns', 'vessel reduction', 'bridge collapse', 'vegetation growth', 'wildfire', 'Wildfire', 'military encampment', 'Assembly', 'equipment withdrawal', 'vehicle damage', 'sedimentation', 'field disturbance', 'event', 'burned land', 'boat gathering', 'crop growth', 'land alteration', 'lava flow', 'Haze', 'Vessel grounding', 'exhibit', 'seasonal display', 'event preparations', 'Campsite development', 'Storm damage', 'landfill search', 'Base dismantlement', 'closure', 'Event preparation', 'festival', 'Temporary structure', 'battle damage', 'vehicle density', 'fortification construction', 'encampment removal', 'drought impact', 'Vegetation fire', 'Mining', 'vessel presence', 'Aircraft activity', 'Increased moisture', 'barrier deployment', 'Land modification', 'Eruption', 'tree felling', 'settlement expansion', 'security operation', 'river discoloration', 'setup', 'water discoloration', 'collapse', 'regatta', 'queue', 'Tornado damage', 'water level drop', 'Industrial damage', 'urban damage', 'cleanup', 'water change', 'vessel arrival', 'launch impact', 'Site clearing', 'Protest', 'Greening', 'Reclamation', 'temporary camp', 'event teardown', 'algae bloom', 'landslide', 'aircraft activity', 'dewatering', 'Fire', 'launch activity', 'covering', 'fortification', 'launch', 'aviation gathering', 'Excavation', 'dumping', 'blast damage', 'emission', 'urban destruction', 'agricultural activity', 'vehicle destruction', 'dam collapse', 'ground disturbance', 'vessel launch', 'strikes', 'vegetation browning', 'vehicle queues', 'impact', 'operational change', 'conflict activity', 'Bombardment', 'water turbidity', 'Destruction', 'Vessel gathering', 'river flow', 'vessel movement', 'tsunami', 'ground covering', 'bridge replacement', 'port disruption', 'river modification', 'naval vessel'}


def build_video_list(imagery_dir, article_id: str) -> List[str]:
    """
    Returns a sorted list of image paths as strings for this article.
    Sorting is by filename; if you want chronological sorting, you can sort by extracted date.
    """
    files = list_image_files_for_article(imagery_dir, article_id)
    files = sorted(files, key=lambda p: p.name)
    return [str(p.as_posix()) for p in files]

def _format_options(options: List[str]) -> str:
    """Format like: 'a, b, c, d, or e.'"""
    return ", ".join(options[:-1]) + f", or {options[-1]}."


def build_event_classification_prompt(video: List[str], correct_label: str, not_visible_type) -> str:
    """
    Rules:
      - exactly 5 options
      - correct label always included
      - 'no event visible' always included and ALWAYS LAST
      - remaining options randomly sampled from ALL_EVENT_TYPES
      - order of other options doesn't matter
    """

    # Ensure unique event types
    all_types = list(set(ALL_EVENT_TYPES))

    options = [correct_label]

    # Remove correct label + no-event from candidate pool
    candidates = [
        t for t in all_types
        if t not in {correct_label, 'no event visible'}
    ]

    # Determine how many random ones we need
    if correct_label == 'no event visible':
        # Need 4 random event types
        random_opts = random.sample(candidates, 3)
        options = random_opts  # correct label goes at end
        options.append(not_visible_type)
    else:
        # Need 3 random event types
        random_opts = random.sample(candidates, 3)
        options.extend(random_opts)

    # Shuffle ONLY the non-"no event visible" options
    random.shuffle(options)

    # Append "no event visible" at the end
    options.append('no event visible')

    # 🔹 Force lowercase
    options = [opt.lower() for opt in options]

    prompt = (
        "This is a sequence of low-resolution, optical satellite images capturing the same location at different times: "
        f"<video>\n"
        "Which of the following classes does this sequence of images belong to? Please answer using only one of the following classes: "
        f"{_format_options(options)}"
    )

    return prompt


def create_event_classification_json(
    labels_path: str | Path,
    imagery_dir: str | Path,
    *,
    dataset_name: str = "skyscraper_gdelt_sentinel",
    task_name: str = "event_classification",
    sensor_name: str = "sentinel",
) -> List[Dict[str, Any]]:
    """
    Creates TeoChat JSON records (list of dicts), one per article.
    Uses your revised_dictionary + image parsing helpers.

    Output keys match your screenshot:
      dataset, lat_lon, timestamp, sensor, conversations, id, task, polygon, video
    """
    articles = revised_dictionary(labels_path)
    records: List[Dict[str, Any]] = []

    for a in articles:
        article_id = str(a["article_id"]).replace(".0", "")
        if a['visible'] is False:
            print('yes')
            correct_label = 'no event visible'
            invisible_type = str(a['event_type'])
        else:
            correct_label = str(a["event_type"])
            invisible_type = None

        # Get images + timestamps from imagery folder
        video = build_video_list(imagery_dir, article_id)
        if not video:
            # Skip articles with no imagery folder / no images
            continue

        num_images, timestamps = get_image_dates_for_article(imagery_dir, article_id)

        # Build repeated lat/lon and sensor arrays to match num_images
        lat = float(a["latitude"])
        lon = float(a["longitude"])
        lat_lon = [[lat, lon] for _ in range(num_images)]
        sensors = [sensor_name for _ in range(num_images)]

        # Human prompt with <video> injected and options updated
        human_prompt = build_event_classification_prompt(video, correct_label, invisible_type)

        record = {
            "dataset": dataset_name,
            "lat_lon": lat_lon,
            "timestamp": timestamps,
            "sensor": sensors,
            "conversations": [
                {"from": "human", "value": human_prompt},
                {"from": "gpt", "value": correct_label},
            ],
            "id": int(article_id) if article_id.isdigit() else article_id,
            "task": task_name,
            "polygon": [""],
            "video": video,
        }

        records.append(record)

    return records


def create_event_detection_json(
    labels_path,
    imagery_dir,
    *,
    dataset_name="skyscraper_gdelt_sentinel",
    task_name="event_detection",
    sensor_name="sentinel",
):
    """
    Creates TeoChat JSON records for binary event visibility.

    Same structure as event classification JSON, but:
      - human prompt asks if event_type is occurring
      - gpt response is "Yes" or "No"
    """

    articles = revised_dictionary(labels_path)
    records = []

    for a in articles:
        article_id = str(a["article_id"]).replace(".0", "")
        event_type = str(a["event_type"]).lower()

        # Determine visibility label
        visible_flag = str(a.get("visible")).strip().lower()
        is_visible = visible_flag in {"true", "t", "1", "yes", "y"}

        gpt_answer = "Yes" if is_visible else "No"

        # Get imagery inputs using your helper functions
        video = build_video_list(imagery_dir, article_id)
        if not video:
            continue

        num_images, timestamps = get_image_dates_for_article( article_id)

        # Repeat lat/lon for each image
        lat = float(a["latitude"])
        lon = float(a["longitude"])
        lat_lon = [[lat, lon] for _ in range(num_images)]

        sensors = [sensor_name for _ in range(num_images)]

        # Build human prompt
        human_prompt = (
            "This is a sequence of images capturing the same location at different times: "
            f"<video>\n"
            f"Is {event_type} occurring in these images? Please answer with Yes or No."
        )

        record = {
            "dataset": dataset_name,
            "lat_lon": lat_lon,
            "timestamp": timestamps,
            "sensor": sensors,
            "conversations": [
                {"from": "human", "value": human_prompt},
                {"from": "gpt", "value": gpt_answer},
            ],
            "id": int(article_id) if article_id.isdigit() else article_id,
            "task": task_name,
            "polygon": [""],
            "video": video,
        }

        records.append(record)

    return records


def create_event_description_json(
    labels_path,
    imagery_dir,
    *,
    dataset_name="skyscraper_gdelt_sentinel",
    task_name="event_description",
    sensor_name="sentinel",
):
    """
    Creates TeoChat JSON records for binary event visibility.

    Same structure as event classification JSON, but:
      - human prompt asks if event_type is occurring
      - gpt response is "Yes" or "No"
    """

    articles = revised_dictionary(labels_path)
    records = []

    for a in articles:
        article_id = str(a["article_id"]).replace(".0", "")
        event_caption = str(a["event_caption"])

        if a["visible"] is False:
            continue

        gpt_answer = event_caption

        # Get imagery inputs using your helper functions
        video = build_video_list(imagery_dir, article_id)
        if not video:
            continue

        num_images, timestamps = get_image_dates_for_article( article_id)

        # Repeat lat/lon for each image
        lat = float(a["latitude"])
        lon = float(a["longitude"])
        lat_lon = [[lat, lon] for _ in range(num_images)]

        sensors = [sensor_name for _ in range(num_images)]

        # Build human prompt
        human_prompt = (
            "This is a sequence of images capturing the same location at different times: "
            f"<video> \n"
            f"Describe what is occurring in these images"
        )

        record = {
            "dataset": dataset_name,
            "lat_lon": lat_lon,
            "timestamp": timestamps,
            "sensor": sensors,
            "conversations": [
                {"from": "human", "value": human_prompt},
                {"from": "gpt", "value": gpt_answer},
            ],
            "id": int(article_id) if article_id.isdigit() else article_id,
            "task": task_name,
            "polygon": [""],
            "video": video,
        }

        records.append(record)

    return records


def create_event_grounding_json(
    labels_path,
    imagery_dir,
    *,
    dataset_name="skyscraper_gdelt_sentinel",
    task_name="event_grounding",
    sensor_name="sentinel",
):
    """
    Creates TeoChat JSON records for event grounding.

    Same structure as event detection JSON, but:
      - human prompt asks what the start and end date of the event is
      - gpt response is "start = start_date, end = end_date" or "the event is not occurring"
    """

    articles = revised_dictionary(labels_path)
    records = []

    for a in articles:
        article_id = str(a["article_id"]).replace(".0", "")
        event_type = str(a["event_type"]).lower()
        start = str(a["event_start_date"])
        end = str(a["event_end_date"])

        # Determine visibility label
        visible_flag = str(a.get("visible")).strip().lower()
        is_visible = visible_flag in {"true", "t", "1", "yes", "y"}

        gpt_answer = f"start = {start}, end = {end}" if is_visible else "the event is not occurring"

        # Get imagery inputs using your helper functions
        video = build_video_list(imagery_dir, article_id)
        if not video:
            print('yes')
            continue


        num_images, timestamps = get_image_dates_for_article(imagery_dir, article_id)

        # Repeat lat/lon for each image
        lat = float(a["latitude"])
        lon = float(a["longitude"])
        lat_lon = [[lat, lon] for _ in range(num_images)]

        sensors = [sensor_name for _ in range(num_images)]

        # Build human prompt
        human_prompt = (
            "This is a sequence of images capturing the same location at different times: "
            f"<video>\n"
            f"If {event_type} is occurring in these images, what is the start date and end date for this event? Please answer with two dates in the format start = YYYY/MM/DD, end = YYYY/MM/DD or if the event is not occurring, answer with the event is not occurring"
        )

        record = {
            "dataset": dataset_name,
            "lat_lon": lat_lon,
            "timestamp": timestamps,
            "sensor": sensors,
            "conversations": [
                {"from": "human", "value": human_prompt},
                {"from": "gpt", "value": gpt_answer},
            ],
            "id": int(article_id) if article_id.isdigit() else article_id,
            "task": task_name,
            "polygon": [""],
            "video": video,
        }

        records.append(record)

    return records

## These two snippets of code below create the EVENT CLASSIFICATION json
# records = create_event_classification_json(
#     labels_path="skyscraper_gdelt_sentinel/labels.csv",
#     imagery_dir="skyscraper_gdelt_sentinel/imagery",
# )
# with open("teochat_event_classification.json", "w") as f:
#     json.dump(records, f, indent=2)


## These two snippets of code below create the EVENT DETECTION json
# records = create_event_detection_json(
#     labels_path="skyscraper_gdelt_sentinel/labels.csv",
#     imagery_dir="skyscraper_gdelt_sentinel/imagery",
# )
# with open("teochat_event_detection.json", "w", encoding="utf-8") as f:
#     json.dump(records, f, indent=2)

## These two snippets of code below create the EVENT DESCRIPTION json
# records = create_event_description_json(
#     labels_path="skyscraper_gdelt_sentinel/labels.csv",
#     imagery_dir="skyscraper_gdelt_sentinel/imagery",
# )
# with open("teochat_event_description.json", "w", encoding="utf-8") as f:
#     json.dump(records, f, indent=2)


## These two snippets of code below create the EVENT GROUNDING json
# records = create_event_grounding_json(
#     labels_path="skyscraper_gdelt_sentinel/labels.csv",
#     imagery_dir="skyscraper_gdelt_sentinel/imagery",
# )
# with open("teochat_event_grounding.json", "w", encoding="utf-8") as f:
#     json.dump(records, f, indent=2)
