"""Educational notes; not a clinical or agricultural diagnosis."""
DISEASE_INFO = {
    'Bacterial Spot': ('Bacterial leaf disease that may cause small dark spots.', 'Avoid overhead watering; clean tools; remove severely affected material.'),
    'Early Blight': ('Disease often associated with brown lesions and concentric rings.', 'Remove infected debris; improve airflow; water near the roots.'),
    'Healthy': ('The model classified this leaf as healthy.', 'Continue monitoring, balanced watering, and good airflow.'),
    'Late Blight': ('Potentially serious disease associated with rapidly expanding dark lesions.', 'Limit leaf wetness and seek local agricultural guidance promptly.'),
    'Leaf Mold': ('Fungal disease associated with yellow patches and mold on leaf undersides.', 'Improve ventilation and reduce prolonged humidity.'),
    'Septoria Leaf Spot': ('Fungal leaf disease that can produce many small circular spots.', 'Remove infected debris and avoid overhead watering.'),
    'Spider Mites Two-spotted Spider Mite': ('Mite damage can cause stippling, discoloration and fine webbing.', 'Inspect leaf undersides and seek locally appropriate pest management guidance.'),
    'Target Spot': ('Leaf disease that can produce brown spots with target-like rings.', 'Improve airflow and remove infected plant debris.'),
    'Tomato Mosaic Virus': ('Viral disease that may cause mottled leaf patterns.', 'Clean tools and avoid transferring plant sap between plants.'),
    'Tomato Yellow Leaf Curl Virus': ('Viral disease that may cause yellowing, curling and stunted growth.', 'Monitor whiteflies and consult local agricultural guidance.'),
}

def get_disease_info(name):
    description, precautions = DISEASE_INFO.get(name, ('No description available.', 'Consult a local agricultural expert.'))
    return {'name': name, 'plant': 'Tomato', 'description': description, 'precautions': precautions}
