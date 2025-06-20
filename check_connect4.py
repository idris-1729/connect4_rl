import numpy as np

def check_connect_4_piece(connect_n: int, piece_array: np.array, placed_coord) -> bool:
    ''' Determines current placement of piece at @placed_coord results in connect4'''
    crawl_axis = np.array([
        [0,1],
        [1,0],
        [1,1],
        [-1,1]
    ])
    def coord_is_inbounds(coord, board) -> bool :
        return (0 <= coord[0] < board.shape[0]) and (0<= coord[1] < board.shape[1])
    
    axis_sums = []
    for crawl_dir in crawl_axis:
        sum_dir = 0
        search_sides = [True, True]
        for n in range(connect_n -1):
            left = placed_coord - (n+1)*crawl_dir
            right = placed_coord + (n+1)*crawl_dir
            for i, side in enumerate([left, right]):
                if coord_is_inbounds(side, piece_array) and search_sides[i]:
                    if piece_array[tuple(side)] == 0:
                        search_sides[i] = False #cut the chain
                    else:
                        sum_dir+= 1
                else:
                    search_sides[i] = False
            if not any(search_sides):
                break
        axis_sums.append(sum_dir)
    return any([s + 1>= connect_n for s in axis_sums])

piece_array = np.array([
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0],
    [0,1,1,0,0,0,0],
    [1,0,1,1,0,0,0],
])
col = 2
column_vals = piece_array[:, col]
if np.all(column_vals == 0):
    y_row = piece_array.shape[0] - 1  # bottom row
else:
    y_row = column_vals.argmax() - 1
print(f'y row is {y_row}')

placed_coord = (y_row, col)
res = check_connect_4_piece(4, piece_array, placed_coord)
print(res)

piece_array[y_row, col] = 1
print(piece_array)