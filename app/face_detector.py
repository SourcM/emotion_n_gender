from mtcnn import MTCNN
import cv2
import numpy as np

# def main():
#     image = cv2.imread('./sample.jpg')

#     predictor_path = './shape_predictor_68_face_landmarks.dat'
#     sp = dlib.shape_predictor(predictor_path)
#     detector = dlib.get_frontal_face_detector()
#     im, _, det_flag= detect_n_crop(image,detector,sp)
#     if det_flag == 1:
#         cv2.imshow('croopped ', im)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

def extract_keypoints(det):
    # k = det[0]['keypoints']
    landmarks = np.zeros((5,2), dtype=int)
    for kk, vv in det[0]['keypoints'].items():
        if kk=='nose':
            landmarks[2] = vv
        elif kk=='mouth_right':
            landmarks[4] = vv
        elif kk=='mouth_left':
            landmarks[3] = vv
        elif kk=='right_eye':
            landmarks[1] = vv
        elif kk=='left_eye':
            landmarks[0] = vv
    
    return landmarks

def crop_img(image, coords5):
    # print('about to crop and extract features')
    #align and crop
    templ = np.array([[83,  96],[171, 95], [127, 162], [90, 208], [166, 206]])
    _,_,Tform,_ = procrustes(templ, coords5, scaling=True, reflection='best')
    _,transmat = get_transmat(Tform)
    outImage = cv2.warpAffine(image, transmat, (256, 256))
    outCoord5 = rotate_coords(coords5, Tform)

    return outImage

def detect_n_crop_dlib(image,detector,sp):

    
    #detector = dlib.cnn_face_detection_model_v1('/home/abukar/Dev/Image_Capture_Scripts/dlib_exp/mmod_human_face_detector.dat')
    #sp = dlib.shape_predictor(predictor_path)
    templ = np.array([[83,  96],[171, 95], [127, 162], [90, 208], [166, 206]])
    
    det_flag = 1
    outCoord5 = None
    outImage = None

    dets = detector(image, 1)
    if len(dets) >= 1:
        #shape = sp(image, dets[0].rect)
        shape = sp(image, dets[0])
        coords, coords5 = shape_to_np(shape)
        _,_,Tform,_ = procrustes(templ, coords5, scaling=True, reflection='best')
        _,transmat = get_transmat(Tform)
        outImage = cv2.warpAffine(image, transmat, (256, 256))
        outCoord5 = rotate_coords(coords5, Tform)
    else:
        det_flag = 0
        #print('No face detected')

    return outImage, outCoord5, det_flag

def detect_n_crop(image,detector):

    
    det = detector.detect_faces(image)
    landmarks = extract_keypoints(det)
    img = crop_img(image, landmarks)
    

    return img

def shape_to_np(shape, dtype="int"):
    default_landmarks = np.array([68, 69, 30, 48, 54])
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((70, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
        # return the list of (x, y)-coordinates
    #coordinates 69 and 70
    c69 = coords[36:41,:]
    c69 = (np.round(np.mean(c69, axis =0)))
    c69 = c69.astype(int)
    coords[68]=c69
    
    c70 = coords[42:47,:]
    c70 = (np.round(np.mean(c70, axis =0)))
    c70 = c70.astype(int)
    coords[69]=c70
    
    subset_coords = coords[default_landmarks,:]
    return coords,subset_coords

def procrustes(X, Y, scaling=False, reflection='best'):
   
    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':
        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0
        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:
        # optimum scaling of Y
        b = traceTA * normX / normY
        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2
        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX
    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX
    
    Zr = np.matmul(Y, T)
    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)
    #transformation values 
    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, tform, Zr

def get_transmat(Tform):
    R = np.eye(3)
    R[0:2,0:2] = Tform['rotation']
    S = np.eye(3) * Tform['scale'] 
    S[2,2] = 1
    t = np.eye(3)
    t[0:2,2] = Tform['translation']
    M = np.dot(np.dot(R,S),t.T).T
    return M,M[0:2,:]

def rotate_coords(coords, Tform):
    T = Tform["translation"]
    R = Tform["rotation"]
    S = Tform["scale"]
    
    a = S*coords
    a = np.matmul(a, R)
    a = a + T
    return a

# if __name__ == '__main__':
#     main()