#!/usr/bin/env python
# coding: utf-8

# Imports
import cv2 as cv
import globals
import numpy as np

# Call function SIFT
def SIFT():
    # Initiate SIFT detector
    SIFT = cv.xfeatures2d.SIFT_create()

    return SIFT

# Call function SURF
def SURF():
    # Initiate SURF descriptor
    SURF = cv.xfeatures2d.SURF_create()

    return SURF

# Call function KAZE
def KAZE():
    # Initiate KAZE descriptor
    KAZE = cv.KAZE_create()

    return KAZE

# Call function BRIEF
def BRIEF():
    # Initiate BRIEF descriptor
    BRIEF = cv.xfeatures2d.BriefDescriptorExtractor_create()

    return BRIEF

# Call function ORB
def ORB():
    # Initiate ORB detector
    ORB = cv.ORB_create()

    return ORB

# Call function BRISK
def BRISK():
    # Initiate BRISK descriptor
    BRISK = cv.BRISK_create()

    return BRISK

# Call function AKAZE
def AKAZE():
    # Initiate AKAZE descriptor
    AKAZE = cv.AKAZE_create()

    return AKAZE

# Call function FREAK
def FREAK():
    # Initiate FREAK descriptor
    FREAK = cv.xfeatures2d.FREAK_create()

    return FREAK

# Call function features
def features(image):
    # Find the keypoints
    #image = cv.imread(filename = 'data/targets/menupan.bmp', flags = cv.IMREAD_GRAYSCALE)
    #image = cv.imread(filename = 'data/targets/menupan.bmp', flags = cv.IMREAD_COLOR)
    #print('image.shape :', image.shape);    #exit()
    #print('image.dtype :', image.dtype);    #exit()
    '''
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY);    
    print('image.shape :', image.shape);    #exit()
    '''
    keypoints = globals.detector.detect(image, None)
    #exit()
    # Compute the descriptors
    keypoints, descriptors = globals.descriptor.compute(image, keypoints)
    
    return keypoints, descriptors

# Call function prints
def prints(keypoints,
           descriptor):
    # Print detector
    print('Detector selected:', globals.detector, '\n')

    # Print descriptor
    print('Descriptor selected:', globals.descriptor, '\n')

    # Print number of keypoints detected
    print('Number of keypoints Detected:', len(keypoints), '\n')

    # Print the descriptor size in bytes
    print('Size of Descriptor:', globals.descriptor.descriptorSize(), '\n')

    # Print the descriptor type
    print('Type of Descriptor:', globals.descriptor.descriptorType(), '\n')

    # Print the default norm type
    print('Default Norm Type:', globals.descriptor.defaultNorm(), '\n')

    # Print shape of descriptor
    print('Shape of Descriptor:', descriptor.shape, '\n')

# Call function matcher

def matcher_kevin(image1,
            image2,
            keypoints1,
            keypoints2,
            descriptors1,
            descriptors2,
            matcher,
            descriptor,
            ratio_thresh):

    if matcher == 'BF':
        # Se descritor for um Descritor de Recursos Locais utilizar NOME
        if (descriptor == 'SIFT') or (descriptor == 'SURF') or (descriptor == 'KAZE'):
            normType = cv.NORM_L2
        else:
            normType = cv.NORM_HAMMING

        # Create BFMatcher object
        BFMatcher = cv.BFMatcher(normType = normType,
                                 crossCheck = True)

        # Matching descriptor vectors using Brute Force Matcher
        matches = BFMatcher.match(queryDescriptors = descriptors1,
                                  trainDescriptors = descriptors2)

        # Sort them in the order of their distance
        matches = sorted(matches, key = lambda x: x.distance)
        
        '''
        # Draw first 30 matches
        globals.output = cv.drawMatches(img1 = image1,
                                        keypoints1 = keypoints1,
                                        img2 = image2,
                                        keypoints2 = keypoints2,
                                        matches1to2 = matches[:30],
                                        outImg = None,
                                        flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        return globals.output
        '''
    elif matcher == 'FLANN':
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1

        index_params = dict(algorithm = FLANN_INDEX_KDTREE,
                            trees = 5)

        search_params = dict(checks = 50)

        # Converto to float32
        descriptors1 = np.float32(descriptors1)
        descriptors2 = np.float32(descriptors2)

        # Create FLANN object
        FLANN = cv.FlannBasedMatcher(indexParams = index_params,
                                     searchParams = search_params)

        # Matching descriptor vectors using FLANN Matcher
        matches = FLANN.knnMatch(queryDescriptors = descriptors1,
                                 trainDescriptors = descriptors2,
                                 k = 2)

        # Lowe's ratio test
        #ratio_thresh = 0.7

        # "Good" matches
        good_matches = []

        '''
        # Filter matches
        print('descriptors1.shape :', descriptors1.shape);  #   (# of kp in query image, descriptor dim of kp in query image). Here (1705, 64)
        print('descriptors2.shape :', descriptors2.shape);  #   (# of kp in train image, descriptor dim of kp in train image). Here (493, 64)
        print('type(matches) :', type(matches));            #   list
        print('len(matches) :', len(matches));              #   # of matches.  Here 1705 which is the # of kp in query image
        #exit()
        #   structure of matches
        #   matches : list of list of matches
        #   matches[0] : sorted list of trainIdx's for queryIdx 0
        #       matches[0][0] : the index of source point whose descriptor is the closet to the index 0 point of query image.
        #       matches[0][1] : the index of source point whose descriptor is the 2nd closet to the index 0 point of query image.
        #       matches[0][k] : the index of source point whose descriptor is the k th closet to the index 0 point of query image.
        #   matches[1] : sorted list of trainIdx's for queryIdx 1
        #       matches[1][0] : the index of source point whose descriptor is the closet to the index 1 point of query image.
        #       matches[1][1] : the index of source point whose descriptor is the 2nd closet to the index 1 point of query image.
        #       matches[1][k] : the index of source point whose descriptor is the k th closet to the index 1 point of query image.


        #print('type(matches[0]) :', type(matches[0]));    exit()
        print('len(matches[0]) :', len(matches[0]));                #   k in the k-Nearest Naver. Here 2.
        print('len(matches[1]) :', len(matches[1]));                #   k in the k-Nearest Naver. Here 2.
        print('len(matches[-1]) :', len(matches[-1]));              #   k in the k-Nearest Naver. Here 2.
        print('type(matches[0][0]) :', type(matches[0][0]));        #   cv.DMatch
        #print('dir(matches[0][0]) :', dir(matches[0][0]));    exit()
        for i in [0, 1, -1]:
            print('\ni :', i)
            for j in range(2):
                print('\tj :', j)
                print('\t\tmatches[i][j].imgIdx) :', matches[i][j].imgIdx);
                print('\t\tmatches[i][j].trainIdx) :', matches[i][j].trainIdx);
                print('\t\tmatches[i][j].queryIdx) :', matches[i][j].queryIdx);
                print('\t\tmatches[i][j].distance) :', matches[i][j].distance);
        #exit()
        #   result of above
        #   i : 0
        #       j : 0                               #   The most closest train KP to 'i' th query KP
        #           matches[i][j].imgIdx : 0
        #           matches[i][j].trainIdx : 4
        #           matches[i][j].queryIdx : 0      #   Note that this is the same as 'i' which means this match result is 'query-centric'  
        #           matches[i][j].distance : 704.9  #   The distance from the descriptor of 'i' th query KP to the descriptor of the most closest train KP  
        #       j : 1                               #   The 2nd closest train KP to i th query KP
        #           matches[i][j].imgIdx : 0
        #           matches[i][j].trainIdx : 366
        #           matches[i][j].queryIdx : 0      #   Note that this is the same as 'i' which means this match result is 'query-centric'  
        #           matches[i][j].distance : 742.9  #   The distance from the descriptor of 'i' th query KP to the descriptor of the 2nd closest train KP.  Note that is larger than that of 'j : 0', which is 704.9.  That is why it is the 2nd closest. 
        #   i : 1
        #       j : 0                               #   The most closest train KP to 'i' th query KP
        #           matches[i][j].imgIdx : 0
        #           matches[i][j].trainIdx : 47
        #           matches[i][j].queryIdx : 1      #   Note that this is the same as 'i' which means this match result is 'query-centric'  
        #           matches[i][j].distance : 662.3  #   The distance from the descriptor of 'i' th query KP to the descriptor of the most closest train KP  
        #       j : 1                               #   The 2nd closest train KP to i th query KP
        #           matches[i][j].imgIdx : 0
        #           matches[i][j].trainIdx : 155
        #           matches[i][j].queryIdx : 1      #   Note that this is the same as 'i' which means this match result is 'query-centric'  
        #           matches[i][j].distance : 683.6  #   The distance from the descriptor of 'i' th query KP to the descriptor of the 2nd closest train KP.  Note that is larger than that of 'j : 0', which is 662.3.  That is why it is the 2nd closest. 
        #   i : -1
        #       j : 0                               #   The most closest train KP to the last query KP
        #           matches[i][j].imgIdx : 0
        #           matches[i][j].trainIdx : 245
        #           matches[i][j].queryIdx : 1704   #   Note that this is the same as 'i' which means this match result is 'query-centric'.  Here 'i' is 1704 which is '# of query KP - 1'  
        #           matches[i][j].distance : 685.4  #   The distance from the descriptor of 'i' th query KP to the descriptor of the most closest train KP  
        #       j : 1                               #   The 2nd closest train KP to i th query KP
        #           matches[i][j].imgIdx : 0
        #           matches[i][j].trainIdx : 477
        #           matches[i][j].queryIdx : 1704   #   Note that this is the same as 'i' which means this match result is 'query-centric'.  Here 'i' is 1704 which is '# of query KP - 1'  
        #           matches[i][j].distance : 711.2  #   The distance from the descriptor of 'i' th query KP to the descriptor of the 2nd closest train KP.  Note that is larger than that of 'j : 0', which is 685.4.  That is why it is the 2nd closest. 
        print('len(keypoints1) :', len(keypoints1));
        print('len(keypoints2) :', len(keypoints2));
        #exit()
        '''

        for m, n in matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
        '''
        # Draw only "good" matches
        globals.output = cv.drawMatches(img1 = image1,
                                        keypoints1 = keypoints1,
                                        img2 = image2,
                                        keypoints2 = keypoints2,
                                        matches1to2 = good_matches,
                                        outImg = None,
                                        flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        return globals.output
        '''
    '''
    print('type(keypoints1) :', type(keypoints1));              #   list      
    print('type(keypoints1[0]) :', type(keypoints1[0]));        #   cv2.KeyPoint      
    print('dir(keypoints1[0]) :', dir(keypoints1[0]));          #   angle, class_id, convert, octave, overlap, pt, response, size
    print('type(keypoints1[0].pt) :', type(keypoints1[0].pt));  #   tupe      
    print('keypoints1[0].pt :', keypoints1[0].pt);              #   (459.1468, 26.2701)     
    '''
    kpts0 = np.array([kp.pt for kp in keypoints1])
    kpts1 = np.array([kp.pt for kp in keypoints2])
    mkpts0 = np.array([keypoints1[good_match.queryIdx].pt for good_match in good_matches])
    mkpts1 = np.array([keypoints2[good_match.trainIdx].pt for good_match in good_matches])
    '''
    print('kpts0.shape :', kpts0.shape);
    print('kpts1.shape :', kpts1.shape);
    print('mkpts0.shape :', mkpts0.shape);
    print('mkpts1.shape :', mkpts1.shape);  
    #if mkpts0.shape[0] > 0:
        #exit()
    '''
    return kpts0, kpts1, good_matches, mkpts0, mkpts1#, ratio_thresh         
    #return keypoints1, keypoints2, good_matches         

def matcher(image1,
            image2,
            keypoints1,
            keypoints2,
            descriptors1,
            descriptors2,
            matcher,
            descriptor):

    if matcher == 'BF':
        # Se descritor for um Descritor de Recursos Locais utilizar NOME
        if (descriptor == 'SIFT') or (descriptor == 'SURF') or (descriptor == 'KAZE'):
            normType = cv.NORM_L2
        else:
            normType = cv.NORM_HAMMING

        # Create BFMatcher object
        BFMatcher = cv.BFMatcher(normType = normType,
                                 crossCheck = True)

        # Matching descriptor vectors using Brute Force Matcher
        matches = BFMatcher.match(queryDescriptors = descriptors1,
                                  trainDescriptors = descriptors2)

        # Sort them in the order of their distance
        matches = sorted(matches, key = lambda x: x.distance)

        # Draw first 30 matches
        globals.output = cv.drawMatches(img1 = image1,
                                        keypoints1 = keypoints1,
                                        img2 = image2,
                                        keypoints2 = keypoints2,
                                        matches1to2 = matches[:30],
                                        outImg = None,
                                        flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        return globals.output

    elif matcher == 'FLANN':
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1

        index_params = dict(algorithm = FLANN_INDEX_KDTREE,
                            trees = 5)

        search_params = dict(checks = 50)

        # Converto to float32
        descriptors1 = np.float32(descriptors1)
        descriptors2 = np.float32(descriptors2)

        # Create FLANN object
        FLANN = cv.FlannBasedMatcher(indexParams = index_params,
                                     searchParams = search_params)

        # Matching descriptor vectors using FLANN Matcher
        matches = FLANN.knnMatch(queryDescriptors = descriptors1,
                                 trainDescriptors = descriptors2,
                                 k = 2)

        # Lowe's ratio test
        ratio_thresh = 0.7

        # "Good" matches
        good_matches = []

        # Filter matches
        for m, n in matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

        # Draw only "good" matches
        globals.output = cv.drawMatches(img1 = image1,
                                        keypoints1 = keypoints1,
                                        img2 = image2,
                                        keypoints2 = keypoints2,
                                        matches1to2 = good_matches,
                                        outImg = None,
                                        flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        return globals.output
