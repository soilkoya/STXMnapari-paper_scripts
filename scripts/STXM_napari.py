# %%

# read stack informaion from hdrfile
import os
import re
def find_hdr_file(directory):
    """
    Returns the path to the unique .hdr file in the specified directory.

    Args:
        directory (str): The path to the directory to search.

    Returns:
        str: The full path to the .hdr file.

    Raises:
        FileNotFoundError: If no .hdr file is found.
        ValueError: If multiple .hdr files are found.
    """
    hdr_files = [f for f in os.listdir(directory) if f.endswith('.hdr')]
    if len(hdr_files) == 1:
        return os.path.join(directory, hdr_files[0])
    else:
        raise ValueError("Multiple or no .hdr files found in the directory.")


def read_hdr(file_path):
    """
    Reads the .hdr file and extracts spatial and spectral parameters.

    Args:
        file_path (str): Path to the .hdr file.

    Returns:
        tuple: A tuple containing:
            - x_step (float): The X-axis step size.
            - y_step (float): The Y-axis step size.
            - energy_list (list of tuple): List of (index, energy_value) pairs.
    """
    with open(file_path, 'r') as file:
        content = file.read()

    # Extract XStep and YStep
    xstep_match = re.search(r"XStep\s*=\s*([0-9.]+);", content)
    ystep_match = re.search(r"YStep\s*=\s*([0-9.]+);", content)

    if not (xstep_match and ystep_match):
        raise ValueError("XStep or YStep not found in the file.")
    
    xstep = float(xstep_match.group(1))
    ystep = float(ystep_match.group(1))

    # Extract Energy values
    image_energy_pattern = re.compile(
        r'(Image\d+_\d+)\s*=\s*{[^}]*?Energy\s*=\s*([\d.]+)', re.DOTALL
    )
    matches = image_energy_pattern.findall(content)

    if not matches:
        raise ValueError("No Energy entries found in the file.")

    # Return energy values paired with their indices
    energy_list = [(i, float(energy_str)) for i, (_, energy_str) in enumerate(matches)]

    return xstep, ystep, energy_list

def save_energy(energy_list, base_dir, textfile_name="energy.txt"):
    """
    Saves the energy_list to the specified base directory.

    Args:
        energy_list (list): List of (index, energy_value) tuples.
        base_dir (str): Destination directory.
        textfile_name (str): Name of the output file.
    """
    # Check if the base directory exists
    if not os.path.exists(base_dir):
        raise ValueError(f"The specified directory {base_dir} does not exist.")
    
    # Define the output file path
    file_path = os.path.join(base_dir, textfile_name)
    
    with open(file_path, 'w') as file:
        file.write("id, eV\n")  # Write header
        for idx, energy in energy_list:
            file.write(f"{idx}, {energy}\n")  # Write each entry
    print(f"Energy list saved to {file_path}") 


# %%   
import os
import re
import numpy as np
import tifffile
import napari

# TIFFスタックとエネルギー情報を読み込む関数
def load_tiff_stack(data_dir, energy_file, xstep, ystep,viewer=None, flip_vertical=True):
    """
    Loads a TIFF image stack from a directory, associates it with energy metadata,
    and displays it in a Napari viewer.

    Args:
        data_dir (str): Path to the directory containing TIFF files.
        energy_file (str): Path to the text file containing energy information.
        xstep (float): Scale/spacing for the X-axis.
        ystep (float): Scale/spacing for the Y-axis.
        viewer (napari.Viewer, optional): Existing Napari viewer instance.
        flip_vertical (bool): Whether to flip images vertically. Defaults to True.
    
    Returns:
        tuple: (viewer, energies, image_stack)
    """
    energy_dict = {}
    
    # Read energy_file
    with open(energy_file, 'r') as f:
        # Skip header
        header = next(f).strip()
        print(header)
        
        for line in f:
            parts = line.strip().split(',') 
            if len(parts) == 2:  
                try:
                    index = int(parts[0])
                    energy = round(float(parts[1]), 1)
                    energy_dict[index] = energy
                except ValueError:
                    print(f"Could not convert to number: {line.strip()}")
            else:
                print(f"Skipping invalid line: {line.strip()}")  

    # Import TIFF files from the directory
    tiff_files = []
    for f in os.listdir(data_dir):
        if f.endswith(('.tif', '.tiff')):
            if re.search(r'\d+', f): 
                tiff_files.append(f)
            else:
                print(f"Excluding: {f} (No numbers found)")


    # import tiff file
    tiff_files = [f for f in os.listdir(data_dir) if f.endswith(('.tif', '.tiff'))]
    
     # Helper function to extract numerical index for sorting
    def extract_number(filename):
        match = re.search(r'(\d+)\.tif', filename)
        return int(match.group(1)) if match else float('inf')  
    
    tiff_files = sorted(tiff_files, key=extract_number)

    energies = []
    filter_tiff_files = []  
    for f in tiff_files:
        file_id = extract_number(f)
        energy = energy_dict.get(file_id, None)  
        
        if energy is not None:
            # If a match is found, add to our final lists
            energies.append(energy) 
            filter_tiff_files.append(f) 
        else:
            print(f"Skipping TIFF {f} (No energy info found for ID {file_id})")

    # Load images into a NumPy array
    image_stack = [tifffile.imread(os.path.join(data_dir, f)) for f in  filter_tiff_files]
    image_stack = np.array(image_stack) 


    # Flip vertically
    if flip_vertical:
        image_stack = image_stack[:, ::-1, :]

    # Display in Napari
    if viewer is None:
        viewer = napari.Viewer()
    
    viewer.add_image(
        image_stack, 
        name="image_stack", 
        metadata={"energies": energies},  
        scale=(ystep,xstep),  
        contrast_limits=None  
    )

    image_layer = viewer.layers["image_stack"]
    energies_list = image_layer.metadata.get("energies", [])

    return viewer, energies_list, image_stack

# %%
import napari
from PyQt5.QtWidgets import QLabel, QLineEdit, QWidget, QVBoxLayout
import numpy as np

def create_element_map(viewer, image_stack, xstep, ystep, element="C", 
                       pre_range=(280, 283), post_range=(297, 300)):
    """
    Generate an element/chemical map by calculating the difference between two energy ranges.
    
    Parameters:
        viewer: napari.Viewer instance.
        image_stack:
        xstep, ystep: Physical scale per pixel (μm).
        element: Name of the element or functional group for the map.
        pre_range, post_range: Energy ranges (eV) to calculate the mean for subtraction.
    """
    energies = viewer.layers["image_stack"].metadata.get("energies", [])
    energies = np.array(energies)

    # Error handling if energy metadata is missing
    if len(energies) == 0:
        print("❌ Error: No energy information found in metadata.")
        return None

    # calculating
    pre_indices = np.where((energies >= pre_range[0]) & (energies <= pre_range[1]))[0]
    post_indices = np.where((energies >= post_range[0]) & (energies <= post_range[1]))[0]

    first_image = np.mean(image_stack[pre_indices], axis=0)
    last_image = np.mean(image_stack[post_indices], axis=0)

    first_image = np.clip(first_image, a_min=0, a_max=None)
    last_image = np.clip(last_image, a_min=0, a_max=None)

    element_distribution = last_image - first_image

    # Add the distribution map 
    layer_name = f"{element} map ({energies[pre_indices[0]]:.1f}-{energies[pre_indices[-1]]:.1f} eV vs {energies[post_indices[0]]:.1f}-{energies[post_indices[-1]]:.1f} eV)"
    element_distribution_image_layer = viewer.add_image(
        element_distribution,
        name=layer_name,
        scale=(ystep, xstep), # Aligning with physical scale
    )

    return element_distribution

# %%
# ROI Extraction Functions
from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer
import numpy as np
import napari

def create_layers(viewer, xstep):
    
    """
    Create Napari Shapes layer and Points layer for ROI.

    Args:
        viewer (napari.Viewer): The active Napari viewer instance.
        xstep (float): The step size for the X-axis (used for scaling/calibration).

    Returns:
        tuple: A tuple containing (shapes_layer, points_layer).
    """

    image_layer = viewer.layers['image_stack']
    # Get scale and translate from the image layer
    scale = np.array(image_layer.scale)         # e.g. (Z, Y, X)
    translate = np.array(image_layer.translate)

    if scale.size == 3:
        xy_scale = tuple(scale[1:])            
        xy_translate = tuple(translate[1:])    
    else:
        xy_scale = tuple(scale)
        xy_translate = tuple(translate)

    """Create Napari Shapes layer and Points layer for ROI"""
    shapes_layer = viewer.add_shapes(name="ROI",
                                      shape_type="polygon",
                                      face_color=[1, 1, 1, 0] ,
                                      scale=xy_scale,
                                      translate=xy_translate,
                                      blending="translucent")
    points_layer = viewer.add_points([], 
                                     name="ROI label", 
                                     face_color='transparent', 
                                     border_color='transparent', 
                                     scale=xy_scale, 
                                     translate=xy_translate)
    points_layer.text = {
        "string": "{label}",
        "color": "black",
        "anchor": "center"
    }
    

    return shapes_layer, points_layer

def compute_polygon_center(polygon):
    """
    Calculate the center coordinates of the ROI.
    """
    return np.mean(polygon, axis=0)

def update_roi_data( event, shapes_layer, points_layer):
    """
    Function to execute when ROI data is modified. Updates labels and colors.
    """
    # Get up to the first 7 ROIs
    roi_data = list(shapes_layer.data)[:7]  #
    centers = [compute_polygon_center(roi) for roi in roi_data]
    
    # Assign labels: 'a', 'b', 'c', ...
    labels = [chr(97 + i) for i in range(len(centers))]  
    
    # Define color palette (Colorblind-friendly colors)
    colors = {
        "a": "#E69F00",  # Orange
        "b": "#56B4E9",  # Sky Blue
        "c": "#009E73",  # Bluish Green
        "d": "#CC79A7",  # Reddish Purple
        "e": "#0072B2",  # Blue
        "f": "#D55E00",  # Vermilion
        "g": "#F0E442"   # Yellow
    }

    print("\nCurrent ROI Data:")
    for i, (roi, center, label) in enumerate(zip(roi_data, centers, labels)):
        print(f"ROI {label}: {roi}\n  中心: {center}")

    # Update Points 
    points_layer.data = np.array(centers)
    points_layer.properties = {"label": labels}

    
   # Set edge colors 
    edge_colors = [colors[label] for label in labels]
    shapes_layer.edge_color =edge_colors

from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QWidget, QFileDialog
import os

def setup_roi_buttons(shapes_layer, points_layer, viewer):
    """Create buttons for ROI operations and add them to the Napari viewer"""
    class ROIButtonsWidget(QWidget):
        def __init__(self):
            super().__init__()
            layout = QVBoxLayout()
            
            self.delete_button = QPushButton("Remove previous ROI")
            self.delete_button.clicked.connect(self.delete_previous_roi)
            layout.addWidget(self.delete_button)
            
            self.reset_button = QPushButton("Reset ROI")
            self.reset_button.clicked.connect(self.reset_roi)
            layout.addWidget(self.reset_button)

            self.load_button = QPushButton("Import ROI")
            self.load_button.clicked.connect(self.load_rois_from_file)
            layout.addWidget(self.load_button)

            self.setLayout(layout)

        def delete_previous_roi(self):
            """Delete the most recently added ROI"""
            if len(shapes_layer.data) > 0:
                shapes_layer.data = shapes_layer.data[:-1]
            else:
                print("No ROI to delete.")
        
        def reset_roi(self):
            """Clear data from both Shapes and Points layers"""
            shapes_layer.data = []
            points_layer.data = []

        def load_rois_from_file(self):
            """Open file dialog to select and load an NPZ file"""
            filename, _ = QFileDialog.getOpenFileName(self, "Select ROI file", "", "NumPy ZIP (*.npz)")
            if filename:
                self.load_shapes_data_to_layer(filename)

        def load_shapes_data_to_layer(self, npz_path):
            """Load ROI coordinate data from the NPZ file into the layer"""
            if not os.path.exists(npz_path):
                print(f"File does not exist: {npz_path}")
                return
            
            # Load NPZ file and extract arrays (roi_a, roi_b, etc.)
            loaded = np.load(npz_path, allow_pickle=True)
            rois = [loaded[key] for key in sorted(loaded.files)]  # roi_a, roi_b, ...
            shapes_layer.data = rois
            shapes_layer.face_color = [1, 1, 1, 0]  
            print(f"Successfully loaded {len(rois)} ROIs.")
            
    # Create the widget 
    roi_buttons_widget = ROIButtonsWidget()
    viewer.window.add_dock_widget(roi_buttons_widget)


def adjust_text_size(viewer, points_layer):
    """Adjust the font size of the text based on the window size"""

    window_size = viewer.window.qt_viewer.size()
    height = window_size.height()
    # Calculate font size
    # Adjust as needed
    raw_size = height * 0.01
    font_size = max(3, min(raw_size, 36)) 

   
    current_text = points_layer.text
    if isinstance(current_text, dict):
        current_text["size"] = font_size
    else:
        current_text = {
            "string": "{label}",
            "color": "black",
            "anchor": "center",
            "size": font_size,
            "outline_width": 2, 
            "outline_color":"#FFFFFF"      
        }
    points_layer.text = current_text
    points_layer.refresh()



def roi_manager(viewer, xstep):
    """
    Set up ROI management in the Napari viewer.

    Args:
        viewer (napari.Viewer): Napari viewer instance.
        xstep (float): Step size for X-axis calibration.

    Returns:
        tuple: A tuple containing (shapes_layer, points_layer).
    """
    shapes_layer, points_layer = create_layers(viewer, xstep)
    
    # update text size
    def update_font_size():
        adjust_text_size(viewer, points_layer)
    
    # Set up a timer to update every second
    timer = QTimer(viewer.window.qt_viewer)
    timer.timeout.connect(update_font_size)
    timer.start(1000)  # 1000ミリ秒 (1秒) ごとに更新

    shapes_layer.events.data.connect(lambda event: update_roi_data(event, shapes_layer, points_layer))
    setup_roi_buttons(shapes_layer, points_layer, viewer)
    
    return shapes_layer, points_layer  

# %%
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon
import napari
from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QSizePolicy
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd

# Define color palette (Colorblind-friendly)
colors = {
        "a": "#E69F00",  # Orange
        "b": "#56B4E9",  # Sky Blue
        "c": "#009E73",  # Bluish Green
        "d": "#CC79A7",  # Reddish Purple
        "e": "#0072B2",  # Blue
        "f": "#D55E00",  # Vermilion
        "g": "#F0E442"   # Yellow
    }


def compute_roi_intensity_mean(roi, image_stack, energies):
    """
    Calculate the mean intensity value within an ROI across an image stack.

    Args:
        roi (numpy.ndarray): Array of coordinates defining the polygon ROI.
        image_stack (numpy.ndarray): 3D image stack (Z, Y, X).
        energies (list or numpy.ndarray): Energy values corresponding to each slice in the stack.

    Returns:
        tuple: (unique_energies, mean_intensities) where mean_intensities is a numpy array.
    """
    all_intensities = []
    all_energies = []

    for i in range(image_stack.shape[0]):
        img = image_stack[i]
        # Note: polygon(y_coords, x_coords)
        rr, cc = polygon(roi[:, 0], roi[:, 1])
        
        valid_mask = (rr >= 0) & (rr < img.shape[0]) & (cc >= 0) & (cc < img.shape[1])
        rr = rr[valid_mask]
        cc = cc[valid_mask]
        
        region_intensity = img[rr, cc]
        all_intensities.extend(region_intensity)
        all_energies.extend([energies[i]] * len(region_intensity))

    unique_energies = np.unique(all_energies)
    means = []
    

    for energy in unique_energies:
        mask = np.array(all_energies) == energy
        means.append(np.mean(np.array(all_intensities)[mask]))
        
    return unique_energies, np.array(means)


def plot_roi_intensity_mean(shapes_layer, image_stack, energies, ax=None):
    """
    Plot the mean intensity values of ROIs against energy values.

    Args:
        shapes_layer (napari.layers.Shapes): Napari layer containing the ROI data.
        image_stack (numpy.ndarray): 3D image data.
        energies (list or numpy.ndarray): Energy values for the x-axis.
        ax (matplotlib.axes.Axes, optional): Existing axes to plot on. Defaults to None.

    Returns:
        matplotlib.figure.Figure: The figure object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    for i, roi in enumerate(shapes_layer.data):
        unique_energies, means= compute_roi_intensity_mean(roi, image_stack, energies)
        
        roi_color = list(colors.values())[i % len(colors)]
        roi_label = chr(97 + i)  # 'a', 'b', 'c'
        ax.plot(unique_energies, means, color=roi_color, label=f"ROI {roi_label}")

    ax.set_xlabel("Energy")
    # Hide Y-axis labels, ticks, and values
    ax.set_yticks([])               
    ax.set_ylabel("")               
    ax.tick_params(axis='y', left=False, labelleft=False)  
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.grid(False)
    fig.subplots_adjust(bottom=0.2)   

    return fig


# Create Qt Widget
class ROIButtonsWidget(QWidget):
    def __init__(self, viewer, shapes_layer, image_stack, energies, on_update_callback=None):
        """
        Custom Qt Widget for plotting ROI spectra within Napari.

        Args:
            viewer (napari.Viewer): Napari viewer instance.
            shapes_layer (napari.layers.Shapes): Layer containing ROI shapes.
            image_stack (numpy.ndarray): 3D image stack data.
            energies (list or numpy.ndarray): Energy values for the x-axis.
            on_update_callback (callable, optional): Function to call when plot is updated.
        """
        super().__init__()
        self.viewer = viewer
        self.shapes_layer = shapes_layer
        self.image_stack = image_stack
        self.energies = energies
        self.on_update_callback = on_update_callback 
        self.setMinimumSize(300, 300)
        
        # Initialize instance variables
        self.roi_dataframe = None
        self.return_fig = None
        self.current_slice_line = None 
        self.current_energy_text = None  
        self.ax = None  
        
        # Create UI components
        self.button = QPushButton("Plot ROI pectra", self)
        self.button.clicked.connect(self.plot_roi_intensity)
        
        # Create FigureCanvas
        self.canvas = FigureCanvas(plt.figure(figsize=(3, 3)))
        
       # Set size
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Setup layout
        layout = QVBoxLayout()
        layout.addWidget(self.button, stretch=0)  
        layout.addWidget(self.canvas, stretch=1) 

        self.setLayout(layout)

        # the vertical line
        self.viewer.dims.events.current_step.connect(self.update_vertical_line)

    def plot_roi_intensity(self):
        if self.current_slice_line:
            self.current_slice_line.remove()
            self.current_slice_line = None
        if self.current_energy_text:
            self.current_energy_text.remove()
            self.current_energy_text = None

        # Clear the figure
        self.canvas.figure.clf()
        self.canvas.figure.set_constrained_layout(True)
        self.ax = self.canvas.figure.add_subplot(111)

        self.return_fig = plot_roi_intensity_mean(self.shapes_layer, self.image_stack, self.energies, ax=self.ax)
        
        # Store ROI data in a DataFrame
        self.roi_dataframe = self.generate_roi_dataframe()

        # Update the vertical line
        self.update_vertical_line()
        self.canvas.draw_idle()
        

        # callback
        if self.on_update_callback is not None:
            self.on_update_callback(self.roi_dataframe, self.return_fig)

    def update_vertical_line(self, event=None):
        """
        Update the vertical line and energy text based on Napari's current slice.

        """
        ax = self.ax
        if ax is None:
            return

       # Convert current slice index to energy value
        current_index = self.viewer.dims.current_step[0]
        if current_index is None or current_index >= len(self.energies):
            return
        current_energy = self.energies[current_index]
        

        # Remove old vertical line
        if self.current_slice_line:
            self.current_slice_line.remove()

        # Draw new vertical line at
        self.current_slice_line = ax.axvline(
            x=current_energy,
            color="black",
            linestyle="--",
            linewidth=1
        )

        

        # Update energy information text display
        if self.current_energy_text:
            self.current_energy_text.remove()
        # Position text at the center of the x-axis
        x_mid = (ax.get_xlim()[1]) 
        self.current_energy_text = ax.text(
            x_mid, -0.2, 
            f"{current_index} |  {current_energy:.2f} eV",
            ha='center', va='top',
            transform=ax.get_xaxis_transform(), 
            fontsize=9,
            color='black'
        )

        self.canvas.draw()
        


    def generate_roi_dataframe(self):
        """
        Collect intensity data from all ROIs and compile them into a single DataFrame.

        Returns:
            pandas.DataFrame: A DataFrame containing 'Energy', 'Mean Intensity', and 'ROI' labels.
        """
        roi_data_list = []

        # Collect data for each ROI
        for i, roi in enumerate(self.shapes_layer.data):
            unique_energies, means = compute_roi_intensity_mean(
                roi, self.image_stack, self.energies
            )
            roi_label = chr(97 + i)  # Labels: 'a', 'b', 'c', ...

            df = pd.DataFrame({
                "Energy": unique_energies,
                "Mean Intensity": means,
                "ROI": roi_label  
            })

            roi_data_list.append(df)

        if roi_data_list:
            return pd.concat(roi_data_list, ignore_index=True)
        else:
            print("No ROI data found.")
            return pd.DataFrame(columns=["Energy", "Mean Intensity", "Std Dev", "ROI"])



def add_roi_controls(viewer, shapes_layer, image_stack, energies):
    """
    Create the ROI control widget.

    Args:
        viewer (napari.Viewer): Napari viewer instance.
        shapes_layer (napari.layers.Shapes): Layer containing ROI shapes.
        image_stack (numpy.ndarray): image data.
        energies (list or numpy.ndarray): Energy values for the x-axis.

    Returns:
        dict: A dictionary containing the latest 'roi_df' and 'fig'.
    """

    latest_data = {"roi_df": None, "fig": None}

    # Callback function: Triggered when the plot button is pressed
    def on_update(roi_df, fig):
        latest_data["roi_df"] = roi_df
        latest_data["fig"] = fig
        print("✅ ROIが更新されました:")
        print(roi_df.head())

    widget = ROIButtonsWidget(viewer, shapes_layer, image_stack, energies, on_update_callback=on_update)
    viewer.window.add_dock_widget(widget, name="ROI Controls")

    return latest_data

# %%
import os
import numpy as np
import string

def save_shapes_data_per_roi(shapes_layer, base_dir, basename="roi_example"):
    """
    Save each ROI shape data into a single compressed NumPy file (.npz).
    Each ROI is stored with a label key (e.g., 'roi_a', 'roi_b').

    Args:
        shapes_layer (napari.layers.Shapes): The Napari layer containing ROI data.
        base_dir (str): The directory where the file will be saved.
        basename (str, optional): The name of the output file. Defaults to "roi_example".

    Returns:
        str: The full path to the saved .npz file.

    Raises:
        ValueError: If the number of ROIs exceeds 26 (limit of a-z labels).
    """
    shapes_data = shapes_layer.data
    num_rois = len(shapes_data)

    # Check if the number of ROIs is within the supported label range (a-z)
    if num_rois > 26:
        raise ValueError("Number of ROIs exceeds 26 (only 'a' through 'z' are supported).")

    # Generate labels ['a', 'b', 'c', ...] based on the number of ROIs
    roi_names = list(string.ascii_lowercase[:num_rois])
    
    # Create a dictionary mapping labels to ROI coordinate data
    named_rois = {f"roi_{name}": roi for name, roi in zip(roi_names, shapes_data)}

    # Define the save path and export as a compressed NPZ file
    save_path = os.path.join(base_dir, f"{basename}.npz")
    np.savez(save_path, **named_rois)
    
    return save_path

# --- Start of Normalization---
# %%
import numpy as np
import pandas as pd
from larch import Group
from larch.xafs import pre_edge

def normalize_roi_pre_edge(roi_df, e0=284, pre1=-4, pre2=0, norm1=8, norm2=16, nnorm=1, pre_post_line=False):
    """
    Apply pre-edge function to ROI data using Larch.

    Args:
        roi_df (pd.DataFrame): ROI DataFrame containing 'Energy', 'Mean Intensity', and 'ROI' columns.
        e0 (float):  default: 284
        pre1 (float): Start of the pre-edge range relative to e0 (default: -4).
        pre2 (float): End of the pre-edge range relative to e0 (default: 0).
        norm1 (float): Start of the normalization range relative to e0 (default: 8).
        norm2 (float): End of the normalization range relative to e0 (default: 16).
        nnorm (int): Order of the polynomial used for normalization (default: 1).
        pre_post_line (bool): If True, adds 'pre_edge_line', 'post_edge_line', and 'mu' to the output (default: False).

    Returns:
        pd.DataFrame: A DataFrame containing the normalized mean intensity and original ROI data.
    """
    # Group data by ROI label
    grouped = roi_df.groupby('ROI')

    processed_results = []  # List to store processed results
    
    for roi_name, group in grouped:
        # Create a Larch Group to store energy and absorption (mu)
        group_larch = Group(energy=np.array(group['Energy'], dtype=float), 
                            mu=np.array(group['Mean Intensity'], dtype=float))
        
        # Apply the pre_edge function
        pre_edge(group_larch, e0=e0, pre1=pre1, pre2=pre2, norm1=norm1, norm2=norm2, nnorm=nnorm)
        
        group['Normalized Mean Intensity'] = group_larch.flat
        
        if pre_post_line:
            group['pre_edge_line'] = group_larch.pre_edge
            group['post_edge_line'] = group_larch.post_edge
            group['mu'] = group_larch.mu
        
        processed_results.append(group)
    
    # Merge all processed groups back into a single DataFrame
    normalize_roi_df = pd.concat(processed_results, ignore_index=True)
    
    return normalize_roi_df

# --- Start of plot spectrum---
# %%
import matplotlib.pyplot as plt
import numpy as np

# Define colors
colors = {
    "a": "#E69F00",  # Orange
    "b": "#56B4E9",  # Sky Blue
    "c": "#009E73",  # Bluish Green
    "d": "#CC79A7",  # Reddish Purple
    "e": "#0072B2",  # Blue
    "f": "#D55E00",  # Vermilion
    "g": "#F0E442"   # Yellow
}

def plot_roi_normalized(normalize_roi_df, 
                        ax=None, 
                        y_margin=0,
                        label_step=2,
                        figsize=(4, 3),
                        text_size=12):
    """
    Plot Normalized Mean Intensity vs. Energy for each ROI.

    Args:
        normalize_roi_df (pandas.DataFrame): DataFrame containing normalized intensity data for each ROI.
        ax (matplotlib.axes.Axes, optional): Existing axes to plot on. If None, a new axes is created.
        y_margin (float): Vertical margin for the y-axis.
        label_step (int): Step for x-axis tick labels.
        figsize (tuple): Figure size (width, height).
        text_size (int): Font size for labels and text.

    Returns:
        matplotlib.figure.Figure: The generated plot figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    y_offset = 0  

    # Get sorted unique ROI labels ('a', 'b', 'c', ...)
    unique_rois = sorted(normalize_roi_df['ROI'].unique())

    for i, roi_label in enumerate(unique_rois):
        # Assign color to each ROI
        roi_color = list(colors.values())[i % len(colors)]

        # Extract data for the specific ROI
        roi_data = normalize_roi_df[normalize_roi_df['ROI'] == roi_label]

        if not roi_data.empty:
            ax.plot(roi_data['Energy'],
                    roi_data['Normalized Mean Intensity'] + y_offset,
                    color=roi_color,
                    label=f"ROI {roi_label}")
            
            # Apply offset to prevent overlapping (waterfall plot effect)
            y_offset += -1

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # Set x-axis ticks at intervals of 1
    xticks = np.arange(np.floor(xmin), np.ceil(xmax) + 1, 1)
    ax.set_xticks(xticks)

    # Show labels only for steps 
    xticklabels = [str(int(tick)) if tick % label_step == 0 else '' for tick in xticks]
    ax.set_xticklabels(xticklabels, fontsize=text_size)

    # Hide Y-axis ticks and labels
    ax.set_yticks([])
    ax.tick_params(axis='y', left=False, labelleft=False)
    ax.set_ylim(ymin, ymax + y_margin)

    # Set axis labels
    ax.set_xlabel("Energy (eV)", fontsize=text_size)
    ax.set_ylabel("Normalized absorbance", fontsize=text_size)

    # Configure legend (positioned outside to the right)
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=text_size, title='')

    # Disable grid
    ax.grid(False)

    return fig

# --- Start of plot OD image---
# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_napari_layers(img, xstep, ystep, image_stack_max=None,
                       shapes_layer=None, points_layer=None, fontsize=12, fontsize_label=7,  
                       unit="μm", title=None, cmap='gray', vmin=0.05, figsize=(3,3)):
    """
    Overlay Napari Shapes and Points layers onto a Matplotlib figure.

    Args:
        img (np.ndarray): 2D image data (Y, X).
        xstep (float): Physical units per pixel along the X-axis.
        ystep (float): Physical units per pixel along the Y-axis.
        image_stack_max (np.ndarray, optional): Max of image stack.
        shapes_layer (napari.layers.Shapes, optional): Napari Shapes of ROIs.
        points_layer (napari.layers.Points, optional): Napari Points of ROIs.
        fontsize (int): Font size for the plot title.
        fontsize_label (int): Font size for the ROI labels.
        unit (str): Physical unit label (e.g., "μm").
        title (str, optional): Title of the figure.
        cmap (str): Colormap for the image.
        vmin (float): Minimum value for imshow intensity scaling.
        figsize (tuple): Figure size in inches (width, height).

    Returns:
        tuple: (fig, ax) Matplotlib figure and axes objects.
    """
    # Define image extent: [left, right, bottom, top]
    extent = [0, img.shape[1] * xstep, img.shape[0] * ystep, 0]

    # vmin/vmax calculation
    current_vmin = vmin
    if image_stack_max is not None:
        vmax = np.nanmax(image_stack_max)
    else:
        vmax = np.nanmax(img)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, cmap=cmap, extent=extent, origin='upper', vmin=current_vmin, vmax=vmax)
    
    # Hide all axis ticks and labels
    ax.axis('off')
    
    if title:
        ax.set_title(title, fontsize=fontsize)

    # Draw Shapes layer (ROIs)
    if shapes_layer is not None:
        for roi, color in zip(shapes_layer.data, shapes_layer.edge_color):
            roi = np.array(roi)
            # Convert pixel coordinates to physical units
            # add a 0.5 * step offset to align the point with the pixel CENTER
            x = roi[:, 1] * xstep + 0.5 * xstep
            y = roi[:, 0] * ystep + 0.5 * ystep
            
            # Close the polygon 
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            ax.plot(x, y, color=color, linewidth=2)
    
    # Draw Points layer (Labels)
    if points_layer is not None:
        labels = points_layer.properties.get("label", [""] * len(points_layer.data))
        for (y0, x0), label in zip(points_layer.data, labels):
            # Calculate label position in physical units
            x = x0 * xstep + 0.5 * xstep
            y = y0 * ystep + 0.5 * ystep
            ax.text(x, y, label,
                    ha='center', va='center', color='black', fontsize=fontsize_label, 
                    bbox=dict(facecolor='white', edgecolor='none', pad=1.0, alpha=0.6))
    
    return fig, ax

# %%
def add_scalebar(ax, xstep, length_um=10, color='white',
                 height_um=2, fontsize=12, pad_um=5, unit='μm', bar_x=0.1, bar_y=0.1):
    """
    Add a scale bar and unit label to a Matplotlib axis.

    Args:
        ax (matplotlib.axes.Axes): The target axes for drawing.
        xstep (float): Physical units per pixel along the X-axis.
        length_um (float): Length of the scale bar in physical units.
        color (str): Color of the bar and text.
        height_um (float): Thickness of the bar in physical units.
        fontsize (int): Font size for the unit label.
        pad_um (float): Padding between the bar and the text label.
        unit (str): Label for the physical unit.
        bar_x (float): X-coordinate for the bar's starting position.
        bar_y (float): Y-coordinate for the bar's starting position.
    """
    va_text = 'top'
    
    # Draw the scale bar
    rect = patches.Rectangle((bar_x, bar_y), length_um, height_um, color=color)
    ax.add_patch(rect)

    # Draw the text label 
    ax.text(bar_x + length_um / 2, bar_y - pad_um,
            f"{length_um} {unit}", ha='center', va=va_text, color=color, fontsize=fontsize)