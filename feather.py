# Params #################################

# im1 should be the high-res image.
# In this case im1 is also the smaller image

im1 = '12m_product'
im2 = '7m_product'

subim_region = 'subim.reg'

pbcor = False

# Script ###################################

imhead(im1,mode='get',hdkey='restfreq')
imhead(im2,mode='get',hdkey='restfreq')

os.system(f'rm -rf {im2}.regrid')
imregrid(imagename=im2,
         template=im1,
         axes=[0, 1],
         output=f'{im2}.regrid')

os.system(f'rm -rf {im2}.regrid.subim')
imsubimage(imagename=f'{im2}.regrid',
           outfile=f'{im2}.regrid.subim',
           region=subim_region)   
os.system(f'rm -rf {im1}.subim')
imsubimage(imagename=im1,
           outfile=f'{im1}.subim',
           region=subim_region)


# This step is only relevant for if one of the images is total power corrected and the other isn't. Requires the primary beam of the corrected image.

if pbcor == True:

    os.system(f'rm -rf {im1}.pb.subim')
    imsubimage(imagename=f'{im1}.pb',    # Take subminage of the primary beam
            outfile=f'{im1}.pb.subim',
            region=subim_region)
    
    os.system(f'rm -rf {im2}.regrid.subim.depb')
    immath(imagename=[f'{im2}.regrid.subim',
                  f'{im1}.pb.subim'],    # multiply with non pbcor image
       expr='IM0*IM1',
       outfile=f'{im2}.regrid.subim.depb')
    
    os.system(f'rm -rf {im1}_{im2}.feather.image')
    feather(imagename=f'{im1}_{im2}.feather.image',
        highres=f'{im1}.subim',
        lowres=f'{im2}.regrid.subim.depb')
else:
    os.system(f'rm -rf {im1}_{im2}.feather.image')
    feather(imagename=f'{im1}_{im2}.feather.image',
        highres=f'{im1}.subim',
        lowres=f'{im2}.regrid.subim')


