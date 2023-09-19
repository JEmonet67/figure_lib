# Creation stimulus circulaire mouvement apparent
total_duration = 114
Ly = 1283
Lx = 1283
transient = 9
duration = 6
intervalle = 3
rayon = 150
x0_2 = np.ceil(((Ly - rayon * 2) / 2) + rayon)  # Coordonnée x du pixel central 2.
y0_2 = np.ceil(((Ly - (rayon * 2) * 3) / 2) + rayon)  # Coordonnée y du pixel central 2.
x0_1 = np.ceil(((Ly - rayon * 2) / 2) + rayon)  # Coordonnée x du pixel central 1
y0_1 = Ly - y0_2  # Coordonnée y du pixel central 1.


name_video = f"white_appMotion_L{Ly}_l{Lx}_D{total_duration}_stim1_{y0_1},{x0_1}_stim2_{y0_2},{x0_2}_radius{rayon}_duration{duration}_intervalle{intervalle}_begin{transient}f.mp4"
# name_video = f"white_appMotion_L{L}_l{l}_D{total_duration}_stim2_{x0_2},{y0_2}_radius{rayon}_duration{duration}_intervalle{intervalle}_begin{transient}f.mp4"
# name_video = f"white_appMotion_L{L}_l{l}_D{total_duration}_stim1_{x0_1},{y0_1}_radius{rayon}_duration{duration}_intervalle{intervalle}_begin{transient}f.mp4"

img_1 = np.zeros(shape=(Ly, Lx, 4), dtype=np.uint8)
for i in range(Ly):
    for j in range(Lx):
        dist = (i - y0_1) * (i - y0_1) + (j - x0_1) * (j - x0_1)
        if np.sqrt(dist) <= rayon:
            img_1[i, j] = (255, 255, 255, 255)
            # img_1[i,j] = (0,0,0,0)

img_2 = np.zeros(shape=(Ly, Lx, 4), dtype=np.uint8)
for i in range(Ly):
    for j in range(Lx):
        dist = (i - y0_2) * (i - y0_2) + (j - x0_2) * (j - x0_2)
        if np.sqrt(dist) <= rayon:
            img_2[i, j] = (255, 255, 255, 255)
            # img_2[i,j] = (0,0,0,0)

for k in range(0, total_duration):
    if k > transient - 1 and k < duration + transient:
        cv2.imwrite(f"{folder_images}frame{k}.jpg", img_1)
    elif k > transient + duration + intervalle - 1 and k < transient + 2 * duration + intervalle:
        cv2.imwrite(f"{folder_images}frame{k}.jpg", img_2)
    else:
        cv2.imwrite(f"{folder_images}frame{k}.jpg", np.zeros(shape=(Ly, Lx, 4), dtype=np.uint8))

hf.images_to_video_moviepy(name_video, folder_images)