import os
import tarfile
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
import rasterio
import yaml

# -----------------------------------------------------------------------------
#                               showfig
# -----------------------------------------------------------------------------
def showfig(data,flag = None,figsize=(5,5),ax=None):
    colorbar = ax is None
    ax = ax or plt.figure(figsize=figsize)
    if flag:
        img = (data & flag) == flag
    else:
        img = data
    plt.imshow(img)
    if colorbar:
        plt.colorbar()
        
# -----------------------------------------------------------------------------
#                         show_single_result
# -----------------------------------------------------------------------------
def show_single_result(image_data, is_ndvi=False):
    '''
    image_data - a single image file response from openeo
    is_ndvi    - set this to true if you have done NDVI calculations
                 (sets a nicer color map)
    EXAMPLE             
    res=connection.load_collection(s2.s2_msi_l2a,
                         spatial_extent=s2.bbox.karlstad_mini_land,
                         temporal_extent=s2.timespans.one_image,
                        bands=['b08','b04']
                        )
    image_data = res.download(format="gtiff")
    show_single_result(image_data)
    
    '''
    with tempfile.TemporaryDirectory() as tmpdirname:
        fname = f"{tmpdirname}/result.tif"
        with open(fname, "wb") as fp:
            fp.write(image_data)
        src = rasterio.open(fname)
        if is_ndvi:
            plt.imshow(src.read(1), cmap='RdYlGn',vmin=-0.8,vmax=0.8)
        else:
            plt.imshow(src.read(1), cmap="pink")
        plt.title(src.tags()["datetime_from_dim"])
    return [src.read()] # To work seamlessly with zipped results
# -----------------------------------------------------------------------------
#                           show_zipped_results
# -----------------------------------------------------------------------------
def show_zipped_results(image_data, is_ndvi=False):
    '''
    image_data - a single image file response from openeo
    is_ndvi    - set this to true if you have done NDVI calculations
                 (sets a nicer color map)
    EXAMPLE             
    res=connection.load_collection(s2.s2_msi_l2a,
                         spatial_extent=s2.bbox.karlstad_mini_land,
                         temporal_extent=s2.timespans.five_images,
                        bands=['b08','b04']
                        )
    image_data = res.download(format="gtiff")
    show_single_result(image_data)
    
    
    '''
    images = []
    with tempfile.TemporaryDirectory() as tmpdirname:
        tar_fname = f"{tmpdirname}/series.tar.gz"
        with open(tar_fname, "wb") as fp:
            fp.write(image_data)

        tar = tarfile.open(tar_fname)
        tar.extractall(tmpdirname)
        tar.close()
        for ifname in sorted(os.listdir(tmpdirname)):
            image_types = [".tif"]
            if not any(image_type in ifname for image_type in image_types):
                continue
            fname = f"{tmpdirname}/{ifname}"
            src = rasterio.open(fname)
            images.append(src.read())
            if is_ndvi:
                plt.imshow(src.read(1), cmap='RdYlGn',vmin=-0.8,vmax=0.8)
            else:
                plt.imshow(src.read(1), cmap="pink")
            plt.title(src.tags()["datetime_from_dim"])
            plt.show()
        return images

# -----------------------------------------------------------------------------
#                               show_result
# -----------------------------------------------------------------------------
def show_result(image_data, is_ndvi=False):
    try:
        return show_single_result(image_data, is_ndvi)
    except:
        pass
    return show_zipped_results(image_data, is_ndvi)

# -----------------------------------------------------------------------------
#                               get_s3_wqsf_flags
# -----------------------------------------------------------------------------
def get_s3_wqsf_flags():
    '''
    You can get these flags from get_collections, but this is a shortcut for
    training purposes.
    
    '''
    wqsf_flags = {}
    here = Path(__file__).parent
    with open(f"{here}/s3_olci_l2wfr.odc-product.yaml", 'r') as stream:
        s3_meta = yaml.safe_load(stream)
      
        for m in s3_meta['measurements']:
            if 'wqsf' in m['name']:
                bits = (m['flags_definition']['data']['values'])
                bitmap = {}
                for b in bits.keys():
                    bitmap[bits[b]] = b 
                    
                wqsf_flags[m['name']] = bitmap
    return wqsf_flags