def extract_features(url):
    # Example feature extraction logic
    features = {}
    
    # Length of the URL
    features['url_length'] = len(url)
    
    # Count of special characters
    special_characters = "!@#$%^&*()_+-=[]{}|;':\",.<>?/"
    features['special_char_count'] = sum(1 for char in url if char in special_characters)
    
    # Count of digits
    features['digit_count'] = sum(1 for char in url if char.isdigit())
    
    # Count of subdomains
    features['subdomain_count'] = url.count('.') - 1  # Assuming subdomains are separated by dots
    
    # Check for the presence of 'http' or 'https'
    features['has_http'] = int(url.startswith('http'))
    
    # Check for the presence of 'www'
    features['has_www'] = int('www' in url)
    
    return features

def preprocess_url(url):
    # Convert the URL into a feature vector
    features = extract_features(url)
    
    # Convert features dictionary to a list or array as needed for the model
    feature_vector = [
        features['url_length'],
        features['special_char_count'],
        features['digit_count'],
        features['subdomain_count'],
        features['has_http'],
        features['has_www']
    ]
    
    return feature_vector