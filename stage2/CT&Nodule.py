import pydicom
import numpy as np
import os
import pandas as pd


def ReadDICOM(data):
  DICOM_path = []
  for fpath, _, fname in os.walk(data):
    for f in fname:
      DICOM_path.append(os.path.join(fpath, f))

  return DICOM_path

def SortDICOM(path):
  DICOM_slice = []
  for p in path:
    ds = pydicom.dcmread(p)
    if hasattr(ds, "ImagePositionPatient"):
      position = ds.ImagePositionPatient
      DICOM_slice.append((position[2], p))
    
  DICOM_slice.sort(key=lambda x: float(x[0])) # sorted by float(x[0])
    
  sorted_path = [s[1] for s in DICOM_slice]
  sorted_path = np.array(sorted_path)
  return sorted_path # numpy array

def ExtractCTparameter(path):
  CTparameter = []
  thin_cut = None
  low_dose = None
  contrasted = None
  for p in path:
    ds = pydicom.dcmread(p)
    kVp = ds.get((0x0018, 0x0060), "N/A")
    kVp = kVp.value if kVp != "N/A" else None
    mAs = ds.get((0x0018, 0x1151), "N/A")
    mAs = mAs.value if mAs != "N/A" else None
    slice_thickness = ds.get((0x0018, 0x0050), "N/A")
    slice_thickness = slice_thickness.value if kVp != "N/A" else None
    pixel_spacing = ds.get((0x0028, 0x0030), "N/A")
    pixel_spacing = pixel_spacing.value if pixel_spacing != "N/A" else None
    reconstruction = ds.get((0x0018, 0x1100), "N/A")
    reconstruction = reconstruction.value if reconstruction != "N/A" else None
    reconstruction_kernel = ds.get((0x0018, 0x1210), "N/A")
    reconstruction_kernel = reconstruction_kernel.value if reconstruction_kernel != "N/A" else None
    manufacturer = ds.get((0x0008, 0x0070), "N/A")
    manufacturer = manufacturer.value if manufacturer != "N/A" else None
    contrast = ds.get((0x0018, 0x0010), "N/A")
    CTparameter.append({
      "kVp": kVp,
      "mAs": mAs,
      "Slice Thickness": slice_thickness,
      "Pixel Spacing": pixel_spacing,
      "Reconstruction Diameter": reconstruction,
      "Reconstruction Kernel": reconstruction_kernel,
      "Manufacturer": manufacturer,
      "Contrast": contrast
    })
  thin_cut = slice_thickness < 1.5 if slice_thickness != "N/A" else None
  low_dose = mAs < 100 if mAs != "N/A" else None
  contrasted = contrast != "N/A" 
  return CTparameter, thin_cut, low_dose, contrasted

def LocalizetheNodule(data, tx, ty): # with coordinates (x, y, z) in mm. 
  path = ReadDICOM(data) 
  CTdata = SortDICOM(path) 
  ds = pydicom.dcmread(CTdata[0])
  ImageOrientation = ds.ImageOrientationPatient
  print("Image Orientation:", ImageOrientation)
  ImagePosition = ds.ImagePositionPatient
  print("Image Position:", ImagePosition)
  tx = 0
  ty = 1

  x = ImagePosition[0] + (tx * ImageOrientation[0] + ty * ImageOrientation[3])
  y = ImagePosition[1] + (tx * ImageOrientation[1] + ty * ImageOrientation[4])
  z = ImagePosition[2] + (tx * ImageOrientation[2] + ty * ImageOrientation[5])

  return (x, y, z)

'''Main'''
data_folders = [
  "/home/lyy/chenMLNovice/data/HW1data/data1", 
  "/home/lyy/chenMLNovice/data/HW1data/data2", 
  "/home/lyy/chenMLNovice/data/HW1data/data3", 
  "/home/lyy/chenMLNovice/data/HW1data/data4"
]
'''
for data in data_folders:
  # read path
  path = ReadDICOM(data) 
  
  # sort path
  CTdata = SortDICOM(path) 
  
  # CT parameters
  CTparameter, thin_cut, low_dose, contrasted = ExtractCTparameter(CTdata)
  print("-" * 60)
  print(f"\nCT Parameters for folder {data}:")
  df = pd.DataFrame(CTparameter)
  print(df)

  print("\nCT Type Analysis:")
  print(f"Thin-cut CT: {'Yes' if thin_cut else 'No'}")
  print(f"Low-dose CT: {'Yes' if low_dose else 'No'}")
  print(f"Contrast CT: {'Yes' if contrasted else 'No'}")
'''

print(LocalizetheNodule(data_folders[0], 1, 1))





