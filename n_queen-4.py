def print_board(board):
    for row in board:
        print(" ".join(str(x) for x in row))
    print()

def is_safe(board, row, col, n):
    for i in range(n):
        for j in range(n):
            if board[i][j] == 1:
                if i == row or j == col or abs(i - row) == abs(j - col):
                    return False
    return True


def solve(board, row, n):
    if row == n:
        print_board(board)
        return

    if 1 in board[row]:
        solve(board, row + 1, n)
        return

    for col in range(n):
        if is_safe(board, row, col, n):
            board[row][col] = 1
            solve(board, row + 1, n)
            board[row][col] = 0


def n_queens():
    n = int(input("Enter N: "))
    board = [[0] * n for _ in range(n)]
    r, c = map(int, input("Enter first queen (row,col) : ").split())
    if not (1 <= r <= n and 1 <= c <= n):
        print("Initial queen position out of bounds.")
        return
    board[r - 1][c - 1] = 1
    print("\nInitial board:")
    print_board(board)
    print("Solutions:\n")
    solve(board, 0, n)

n_queens()


        
              
