# Import dependencies
import pandas as pd
import os
import base64
from openai import OpenAI
import sys as sys

#Load data

cases = []

# Load case data from organized directory structure
# Each case follows this standardized organization:
# Cases/XX/             # Note: Actual structure uses number folders (e.g., 19, 20, etc.), which align with idnum on case index
# ├── CT/
# │   ├── Axial/
# │   │   ├── bone_window/
# │   │   ├── C+_arterial_phase/
# │   │   └── non-contrast/
# │   ├── Coronal/
# │   │   ├── bone_window/
# │   │   ├── C+_arterial_phase/
# │   │   └── non-contrast/
# │   └── Sagittal/
# │       ├── bone_window/
# │       ├── C+_arterial_phase/
# │       └── non-contrast/
# └── MRI/
#     ├── Axial/
#     │   ├── ADC/
#     │   ├── DWI/
#     │   ├── FLAIR/
#     │   ├── Gradient_Echo/
#     │   ├── STIR/
#     │   ├── SWI/
#     │   ├── T1/
#     │   ├── T1_C+/
#     │   ├── T1_C+_fat_sat/
#     │   ├── T2/
#     │   └── T2_fat_sat/
#     └── Coronal/
#         ├── ADC/
#         ├── DWI/
#         ├── FLAIR/
#         ├── Gradient_Echo/
#         ├── STIR/
#         ├── SWI/
#         ├── T1/
#         ├── T1_C+/
#         ├── T1_C+_fat_sat/
#         ├── T2/
#         └── T2_fat_sat/
#
# The code walks through this structure and creates image labels by concatenating
# the subdirectory names (e.g., "CT_Axial_bone_window" or "MRI_Coronal_T1_C+")

case_path = r"PATH_TO_YOUR_DATA\Cases"

# Load case index CSV file
# The case_index DataFrame should contain the following columns:
# - idnum: Unique case identifier (matches folder names in Cases directory)
# - Diagnosis: Medical diagnosis description
# - Presentation: Clinical presentation/symptoms
# - Age: Patient age
# - Gender: Patient gender (M/F)
# Ensure the CSV file matches this structure for proper data loading

case_index = pd.read_csv(r"PATH_TO_YOUR_CASE_INDEX")

for case in os.listdir(case_path):
    case_folder = os.path.join(case_path, case)
    if os.path.isdir(case_folder):
        # Get case information from case_index DataFrame
        case_info = case_index[case_index['idnum'] == int(case)]
        if len(case_info) > 0:
            gender = 'male' if case_info['Gender'].iloc[0] == 'M' else 'female'
            clinical_info = f"{case_info['Age'].iloc[0]} year old {gender}. {case_info['Presentation'].iloc[0]}"
        else:
            clinical_info = 'Unknown'
        
        images = []
        # Walk through all subdirectories
        for root, dirs, files in os.walk(case_folder):
            if files:  # Only process if there are files in this directory
                for file in files:
                    full_file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(root, case_folder)
                    label = '_'.join(rel_path.split(os.sep))
                    images.append({
                        "path": full_file_path,
                        "label": label
                    })
                    
        if images:  # Only add the case if it has files
            cases.append({
                "case_id": case,
                "diagnosis": case_info['Diagnosis'].iloc[0],
                "clinical_info": clinical_info,
                "images": images
            })

#Encode images for model processing
client = OpenAI(api_key=PUT_YOUR_OPENAI_KEY_HERE)

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def create_file(client, file_path):
    with open(file_path, "rb") as file_content:
        result = client.files.create(
            file=file_content,
            purpose="assistants"
        )
    return result.id

# Encode reference text for model processing
file_id = create_file(client, PATH_TO_ACADEMIC_TEXTS_HERE)

vector_store = client.vector_stores.create(
    name="knowledge_base"
)

result = client.vector_stores.files.create(
    vector_store_id=vector_store.id,
    file_id=file_id
)

#Define prompts and models
prompts = {
    'simple' : 
    """You are a specialized neuroradiology AI assistant trained in orbital imaging interpretation. Based on the clinical presentation, and the CT and/or MRI images provided, provide the most likely diagnosis and four main differentials for the orbit pathology. Use the following format for your response:

    **Primary Diagnosis**: PRIMARY DIAGNOSIS HERE

    **Differentials**:
    1. FIRST DIFFERENTIAL
    2. SECOND DIFFERENTIAL
    3. THIRD DIFFERENTIAL
    4. FOURTH DIFFERENTIAL""", 

    'complex' :
    """You are a specialized neuroradiology AI assistant trained in orbital imaging interpretation. Please consider the following for each case: Is the lesion located in the optic nerve sheath complex, intraconal space, conal space or extraconal compartments? Integrate the clinical presentation (e.g., age, symptoms, laterality, acuity of onset) with the imaging findings (e.g., lesion density/signal, enhancement characteristics, margins, associated features like proptosis or bony remodelling) and provide the most likely diagnosis, as well as four main differentials. Use the following format for your response:

    **Primary Diagnosis**: PRIMARY DIAGNOSIS HERE

    **Differentials**:
    1. FIRST DIFFERENTIAL
    2. SECOND DIFFERENTIAL
    3. THIRD DIFFERENTIAL
    4. FOURTH DIFFERENTIAL"""
}

models = ['o3-2025-04-16', 'gpt-5-2025-08-07']

#Test each model with each prompt
for key, prompt_text in prompts.items():
    for model in models:
        print(f"Testing {key} prompt with {model}...")
        prompt_text = prompt_text

        model = model

        answers = []
        system = prompt_text

        final_df = { 
            'case_id': [],
            'diagnosis': [],
            'answers': []
        }

        for case in cases:
            print(f"Processing case {case['case_id']}")
            content = [
            ({"type":"input_text", "text":f'Presentation: {case['clinical_info']}'})]

            for img in case['images']:
                content.append({
                    "type": "input_text",
                    "text": img['label']
                })
                content.append({
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{encode_image_to_base64(img['path'])}"
                })

            response = client.responses.create(
                model= model, 
                input=[{"role": "system", "content": system}, {"role": "user", "content": content}], 
            )

            response_text = response.output_text

            final_df['case_id'].append(case['case_id'])
            final_df['diagnosis'].append(case['diagnosis'])
            final_df['answers'].append(response_text)

        # Convert dictionary to DataFrame
        results_df = pd.DataFrame(final_df)

        #Save to CSV file
        output_path = fr"PATH_TO_RESULTS\{model} {key}.csv"
        results_df.to_csv(output_path, index=False)


#Test the GPTV model with the file search tool

prompt_text = prompts['simple']
model = models[1]
tool = [{"type": "file_search", "vector_store_ids":[vector_store.id]},]

print(f"Testing simple prompt with {model} and file search...")

answers = []
system = prompt_text

final_df = { 
    'case_id': [],
    'diagnosis': [],
    'answers': []
}

for case in cases:
    print(f"Processing case {case['case_id']}")
    content = [
    ({"type":"input_text", "text":f'Presentation: {case['clinical_info']}'})]

    for img in case['images']:
        content.append({
            "type": "input_text",
            "text": img['label']
        })
        content.append({
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{encode_image_to_base64(img['path'])}"
        })

    response = client.responses.create(
        model= model, 
        input=[{"role": "system", "content": system}, {"role": "user", "content": content}], 
        tools=tool, 
        tool_choice='required',
    )

    response_text = response.output_text

    final_df['case_id'].append(case['case_id'])
    final_df['diagnosis'].append(case['diagnosis'])
    final_df['answers'].append(response_text)

# Convert dictionary to DataFrame
results_df = pd.DataFrame(final_df)

#Save to CSV file
output_path = fr"PATH_TO_YOUR_RESULTS\{model} file search.csv"
results_df.to_csv(output_path, index=False)
