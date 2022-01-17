import cv2
from skimage import exposure
from constants import *
from utils import *

file_loc = './data/Original/Images/'

create_folder("./data/training_data/")

for img_det in img_details:
    file_name = img_det['name']
    grid = img_det['grid']
    tl = img_det['tl']
    tr = img_det['tr']
    bl = img_det['bl']
    br = img_det['br']
    path = os.path.join(file_loc, file_name)
    orig = cv2.imread(path)
    if file_name == 'DSC02086.JPG':
        orig = cv2.copyMakeBorder(orig, top=0, right=0, left=200, bottom=0, borderType=cv2.BORDER_CONSTANT,
                                  value=[0, 0, 0])

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    rect = np.array([list(tl), list(tr), list(br), list(bl)], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
    warp = exposure.rescale_intensity(warp, out_range=(0, 255))
    cv2.imwrite('./data/training_data/' + file_name[:-4] + '_cropped_' + img_det['set'] + file_name[-4:], warp)
    warp_grid = warp.copy()
    h, w, d = warp.shape
    for i in range(1, grid[0]):
        val = int(h / grid[0] * i);
        cv2.line(warp_grid, (0, val), (w - 1, val), (0, 255, 0), 5)
    for i in range(1, grid[1]):
        val = int(w / grid[1] * i);
        cv2.line(warp_grid, (val, 0), (val, h - 1), (0, 255, 0), 5)
    cv2.imwrite('./data/training_data/' + file_name[:-4] + '_grid_' + img_det['set'] + file_name[-4:], warp_grid)
    if img_det['form'] == 1:
        slab = 0
    else:
        slab = 0.2

    w_grid = float(w) / grid[1]
    w_slab = w_grid * slab
    h_grid = float(h) / grid[0]
    h_slab = h_grid * slab
    print(w_grid, h_grid)
    print(file_name)
    scale = max(w_grid, h_grid, 100)
    scale = scale / 100
    print(scale)
    ct = 1
    folder_name = './data/training_data/' + file_name[:-4] + '_' + img_det['set'] + '/'
    create_folder(folder_name)

    for i in range(grid[0]):
        for j in range(grid[1]):
            x = int(j * w_grid)
            y = int(i * h_grid)
            cv2.putText(warp_grid, str(ct), (x + int(w_grid / 2), y + int(h_grid / 2)), cv2.FONT_HERSHEY_SIMPLEX, scale,
                        (0, 0, 255), 5)
            img_section = warp[y + int(h_slab):y + int(h_grid) - int(h_slab),
                          x + int(w_slab):x + int(w_grid) - int(w_slab)]
            cv2.imwrite(folder_name + file_name[:-4] + '_' + img_det['set'] + '_' + str(ct) + '.jpg', img_section)
            ct += 1
    cv2.imwrite('./data/training_data/' + file_name[:-4] + '_grid_' + img_det['set'] + file_name[-4:], warp_grid)
