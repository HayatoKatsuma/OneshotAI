import json
import sys
import neoapi

def get_all_camera_parameters():
    """
    Connects to the first available Baumer camera, reads all readable
    parameters, and returns them as a dictionary.
    """
    camera = None
    try:
        # Connect to the first available camera
        camera = neoapi.Cam()
        camera.Connect()
        print(f"Connected to: {camera.f.DeviceModelName.GetString()} ({camera.f.DeviceSerialNumber.GetString()})", file=sys.stderr)

        parameters = {}
        
        # Get the list of all available features
        feature_list = camera.GetFeatureList()

        print(f"Found {feature_list.GetSize()} features. Reading readable ones...", file=sys.stderr)

        for feature_info in feature_list:
            # Skip features that are not readable
            if not feature_info.IsReadable():
                continue

            name = feature_info.GetName()
            
            try:
                # Get the feature accessor object from the camera instance
                feature_obj = getattr(camera.f, name)
                
                value = None
                # Determine the type of the feature to read its value correctly.
                # Based on examples, Enumeration types have GetString/SetString methods,
                # while others (int, float, bool) use the .value property.
                if hasattr(feature_obj, 'GetString'):
                    value = feature_obj.GetString()
                elif hasattr(feature_obj, 'value'):
                    value = feature_obj.value
                else:
                    # This could be a category node or a command, which we can't read a single value from.
                    continue
                
                parameters[name] = value

            except (neoapi.NeoException, AttributeError):
                # Silently ignore features that cannot be read for any reason
                pass
        
        return parameters

    except neoapi.NeoException as e:
        print(f"Camera Error: {e}", file=sys.stderr)
        return None
    finally:
        if camera and camera.IsConnected():
            camera.Disconnect()
            print("Camera disconnected.", file=sys.stderr)

if __name__ == "__main__":
    import os
    # Get the parameters
    params = get_all_camera_parameters()
    
    if params:
        # Define the output path
        output_dir = "config"
        output_file = "now_camera_params.json"
        output_path = os.path.join(output_dir, output_file)

        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save the parameters to the JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=4)
        
        print(f"Successfully saved camera parameters to {output_path}")
    else:
        print("Failed to retrieve camera parameters.", file=sys.stderr)
        sys.exit(1)
