import numpy as np
from astropy.convolution import convolve,convolve_fft, Box2DKernel
import skimage.measure    
from stackarator.dist_ellipse import dist_ellipse
import astropy.units as u
from skimage.transform import rotate
import warnings
from astropy.utils.exceptions import AstropyUserWarning


class cas_molecules:
    def __init__(self,image,inc,pa,xc,yc,cellsize,scale=0):
        self.image=image
        self.inc=inc
        self.pa=pa
        self.cellsize=cellsize
        self.xc=xc
        self.yc=yc
        self.smooth_scale=scale
        self.rad_array=dist_ellipse(self.image.shape, self.xc, self.yc, 1/np.cos(np.deg2rad(self.inc)), pa=self.pa+90) # (Dan) This is building the elliptical aperture
        
        ### create flux vs radius
        flatim=self.image.flatten()
        flatrad=self.rad_array.flatten() 
        s=np.argsort(flatrad)
        flatim=flatim[s]
        self.flatrad=flatrad[s]
        self.arr=np.nancumsum(flatim)/np.nansum(image)
            
    def calc_radius(self,frac):
        return np.interp(frac,self.arr,self.flatrad) # (Dan) Finds the 90% flux radius by flattening the image and rad_array, sorting by radius, and then finding the radius at which the cumulative flux reaches the desired fraction.
    
    
    def gini(self):
        """Calculate the Gini coefficient of a numpy array."""
        array2 = np.float64(self.image)  # skimage wants double
        r80s=self.calc_radius([0.85,0.9,0.95])
        ginis=[]
        for r80 in r80s:
            array=array2.copy()
            array=array[np.isfinite(array)&(self.rad_array<r80)] # (Dan) not a boolean mask in the way we've been doing
            array = array.flatten()
            if np.nanmin(array) < 0:
                #Values cannot be negative:
               array -= np.amin(array) # (Dan) boost the array so that the minimum value is 0.
            # Values cannot be 0:
            array += 0.0000001 # (Dan) boost the array a bit more so that the minimum value is not 0.
            # Values must be sorted:
            array = np.sort(array)
            # Index per array element:
            index = np.arange(1,array.shape[0]+1)
            # Number of array elements:
            n = array.shape[0]
            # Gini coefficient:
            ginis.append(((np.nansum((2 * index - n  - 1) * np.abs(array))) / (n * np.nansum(np.abs(array)))))
            # my gini: 1/(mean*n*(n-1)) * np.sum((2*index - n - 1) * sorted_vals)       (Dan) This this Gini formula is the same. array is same as sorted_vals. mean = nansum(array)/(n-1)

        return ginis[1],np.sqrt(np.prod(np.abs(ginis[1]-[ginis[0],ginis[2]])))#(np.max(ginis)-np.min(ginis))/2.   # (Dan) Errors are calculated by by going ±5% in the the flux aperture.
        
        

    def m20(self):
         """
         Calculate the M_20 coefficient as described in Lotz et al. (2004).
         """

         # Use the same region as in the Gini calculation
         image = np.float64(self.image)  # skimage wants double

         # Calculate second total central moment
         Mc = skimage.measure.moments_central(np.nan_to_num(image), center=(self.yc, self.xc), order=2)
         second_moment_tot = Mc[0, 2] + Mc[2, 0]

         # Calculate threshold pixel value
         sorted_pixelvals = np.sort(image.flatten())
         flux_fraction = np.nancumsum(sorted_pixelvals) / np.nansum(sorted_pixelvals)
         sorted_pixelvals_20 = sorted_pixelvals[flux_fraction >= 0.8]
         if len(sorted_pixelvals_20) == 0:
             # This can happen when there are very few pixels.
             warnings.warn('[m20] Not enough data for M20 calculation.',
                           AstropyUserWarning)
             return -99.0  # invalid
         threshold = sorted_pixelvals_20[0]

         # Calculate second moment of the brightest pixels
         image_20 = np.where(image >= threshold, image, 0.0)
         Mc_20 = skimage.measure.moments_central(np.nan_to_num(image_20), center=(self.yc, self.xc), order=2)
         second_moment_20 = Mc_20[0, 2] + Mc_20[2, 0]

         if (second_moment_20 <= 0) | (second_moment_tot <= 0):
             warnings.warn('[m20] Negative second moment(s).',
                           AstropyUserWarning)
             m20 = -99.0  # invalid
         else:
             m20 = np.log10(second_moment_20 / second_moment_tot)
         return m20   

    def concentration(self):
        image = np.float64(self.image)  # skimage wants double
        xv, yv = np.meshgrid(np.arange(0,image.shape[0]), np.arange(0,image.shape[1]))
        rad2=(xv-self.xc)**2 + (yv-self.yc)**2
        s=np.argsort(rad2.ravel())
        arr=np.nancumsum((image.ravel())[s])/np.nansum(image)
        r20=np.sqrt(np.interp(0.2,arr,rad2.ravel()))
        r80=np.sqrt(np.interp(0.8,arr,rad2.ravel()))
        return 5*np.log10(r80/r20)
    
    
    def asym(self):
        image = np.float64(self.image)  # skimage wants double
        r80s=self.calc_radius([0.85,0.9,0.95])
        asyms=[]
        for r80 in r80s:
            image1=image.copy()
            image1[self.rad_array>r80]=0 # (Dan) Anything outside f90 radius is set to 0
            imagerot=rotate(image1,180,center=(self.xc,self.yc))
            asyms.append(np.nansum(np.abs(image1-imagerot))/np.nansum(np.abs(image1))) # (Dan) yeah this is the same
        return asyms[1],np.sqrt(np.prod(np.abs(asyms[1]-[asyms[0],asyms[2]])))
        

    def smoothness(self):
        image = np.float64(self.image)  # skimage wants double
        r80s=self.calc_radius([0.85,0.9,0.95])
        smooths=[]
        for r80 in r80s:
            image1=image.copy()
            image1[self.rad_array>r80]=0 # (Dan) again, anything outside f90 radius is set to 0
            #smoothed_image = convolve_fft(image, Box2DKernel(np.round(image.shape[0]/6.)))
            smoothed_image = convolve_fft(image1, Box2DKernel(np.round(self.smooth_scale))) # (Dan) using astropy convolve 2d box
            resid=image1-smoothed_image
            smooths.append(np.nansum(np.abs(resid[(resid>0)]))/np.nansum(np.abs(image1))) # (Dan) rejects negative diff
            
        
        #smoothed_image = convolve_fft(image, Box2DKernel(np.round(self.smooth_scale)))
        #resid=image1-smoothed_image
        #smooths[1]=(np.nansum(np.abs(resid[resid>0]))/np.nansum(np.abs(image)))
                    
        
            
        return smooths[1],np.sqrt(np.prod(np.abs(smooths[1]-[smooths[0],smooths[2]])))     
        
    def run_all(self,clip,gasden_clip_rad_pix=np.inf):
        
        r90=self.calc_radius(0.9)*self.cellsize
        (g,egini),m_20,c,(a,ea),(s,es)=self.gini(),self.m20(),self.concentration(),self.asym(),self.smoothness()
        
       # breakpoint()
        gasden_clip_rad_pix = self.calc_radius(0.9)
        areatoadd =np.pi*np.cos(np.deg2rad(self.inc))*(gasden_clip_rad_pix**2)- (self.rad_array<gasden_clip_rad_pix).sum()
        if areatoadd >0:
             meangasden=np.nanmean(np.append(self.image[(self.rad_array<gasden_clip_rad_pix)],np.zeros(int(areatoadd))))
        else:
            meangasden=np.nanmean(self.image[(self.rad_array<gasden_clip_rad_pix)])
        
        meangasden_nozero=np.nanmean(self.image[(self.rad_array<gasden_clip_rad_pix)&(self.image>0)])
        
        if not np.isfinite(meangasden):
            breakpoint()
        
        #print("M20=",m_20)
        #print("C=",c)
        
        print('\nDavis method\n')
        print("r90=",r90/1000,"kpc")
        print("S=",s,"±",es)
        print("A=",a,"±",ea)
        print("Gini=",g,"±",egini)
    
        return g,m_20,c,a,s,meangasden,egini,ea,es,meangasden_nozero,r90
        
        
        
            