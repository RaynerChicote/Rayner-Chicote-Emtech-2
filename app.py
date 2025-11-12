def get_disease_info(disease_name):
    """Get information about the detected disease"""
    # Clean the disease name
    disease_name = disease_name.lower().replace('_', ' ').title()
    
    disease_info = {
        "Brown Spot": {
            "symptoms": "Small, oval brown spots on leaves, may have yellow halo",
            "treatment": "Use resistant varieties, proper fertilization, fungicide application",
            "prevention": "Crop rotation, proper field sanitation"
        },
        "Bacterial Leaf Blight": {
            "symptoms": "Water-soaked lesions that turn yellow and then white",
            "treatment": "Copper-based bactericides, antibiotic sprays",
            "prevention": "Use clean seeds, avoid waterlogged conditions"
        },
        "Leaf Smut": {
            "symptoms": "Black powdery masses on leaves, reduced plant vigor",
            "treatment": "Fungicide application, remove infected plants",
            "prevention": "Crop rotation, field sanitation"
        },
        "Healthy": {
            "symptoms": "No visible symptoms, green and vibrant leaves",
            "treatment": "Maintain current practices",
            "prevention": "Regular monitoring, proper nutrition"
        }
    }
    
    # Try exact match first
    if disease_name in disease_info:
        return disease_info[disease_name]
    
    # Try to find the closest match
    for key in disease_info.keys():
        if any(word in disease_name.lower() for word in key.lower().split()):
            return disease_info[key]
    
    return {
        "symptoms": "Information not available",
        "treatment": "Consult agricultural expert",
        "prevention": "Regular monitoring and proper cultivation practices"
    }
