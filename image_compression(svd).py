from PIL import Image
import numpy as np


image = Image.open("Images/skull.jpg")
data = np.array(image)

def  intro(sigma):
    length = len(sigma)
    print("=== Image Compression using SVD ===\n")
    print("NOTE:")
    print("• More singular values → Better image quality (Less compression)")
    print("• Fewer singular values → Lower image quality (More compression)\n")

    print(f"Total singular values available: {length}")

    num = int(input(f"Enter the number of singular values to keep (1 to {length}): "))

    if num > length:
        print("\n !!! You entered a number greater than the available singular values.")
        print("Please run again and choose a valid number.")
        exit()
    if num <= 0:
        print("\n !!! You entered a number lesser than the Range singular values.")
        print("Please run again and choose a valid number.")
        exit()
    else:
        print(f"\nUsing top {num} singular values to compress the image...\n")

    return num


if data.ndim == 2:
    u, sigma, vt = np.linalg.svd(data)
    num = intro(sigma)
    sigma = sigma[:num]
    sigma = np.diag(sigma)
    new = u[:,:num] @ sigma @ vt[:num,:]
    new = np.clip(new, 0, 255).astype(np.uint8)
    newimg = Image.fromarray(new)
    newimg.show()
else:
    if data.ndim == 3:
        u1, sigma1, vt1 = np.linalg.svd(data[:, :, 0])
        u2, sigma2, vt2 = np.linalg.svd(data[:, :, 1])
        u3, sigma3, vt3 = np.linalg.svd(data[:, :, 2])
        k = len(sigma1)
        num = intro(sigma1)
        sigma1 = sigma1[:num]
        sigma1 = np.diag(sigma1)
        sigma2 = sigma2[:num]
        sigma2 = np.diag(sigma2)
        sigma3 = sigma3[:num]
        sigma3 = np.diag(sigma3)
        #1.in top k svd, sigma matrix should not be equal to original matrix size (M X N)
        #2.it should be K X K (square mat) because M X K @ K X K @ K X N
        r = u1[:,:num] @ sigma1 @ vt1[:num,:]
        g = u2[:, :num] @ sigma2 @ vt2[:num, :]
        b = u3[:, :num] @ sigma3 @ vt3[:num, :]
        r = np.clip(r,0,255).astype(np.uint8)
        g = np.clip(g, 0, 255).astype(np.uint8)
        b = np.clip(b, 0, 255).astype(np.uint8)
        singlearray = np.stack([r,g,b],axis = 2)
        img = Image.fromarray(singlearray)
        img.save(f'Outputs/compressed_k={num}.jpg')



#concept of building this:
#svd = U@Σ@V.T
#svd of mat = rotation @ scaling @ rotation
#scaling (sigma mat) - contain singular vals(√λ) it describe strength (or) significance of direction(vector) it is number it scales respective vector in u and v.T
#so the bigger scalar the important the vector is
#finally we are taking the required scalar rows and building the matrix back again , sigma mat is given in descending ord
#so the more scalr rows you take better the image
#less scalar values - more compression - less quality