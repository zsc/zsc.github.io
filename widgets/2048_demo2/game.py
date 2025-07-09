# game.py
import numpy as np
import random
from typing import Tuple, List
import unittest

class Game2048:
    """
    无状态的2048操作集合：
    - 棋盘用一个64位整数表示，每4位为一个格子的 log2(tile)。
    - 通过类方法来移动、增加随机格、检测结束等。
    """

    # 类级查找表，第一次调用时自动初始化
    _left_table    = {}
    _right_table   = {}
    _score_table   = {}
    _transpose_tbl = {}
    _initialized   = False

    @classmethod
    def _init_tables(cls):
        if cls._initialized:
            return
        cls._initialized = True

        def int_to_row(x:int) -> List[int]:
            return [(x >> (4*i)) & 0xF for i in range(4)]
        def row_to_int(row:List[int]) -> int:
            out = 0
            for i,v in enumerate(row):
                out |= (v << (4*i))
            return out
        def move_row_left(row:List[int]) -> Tuple[List[int],int]:
            nz = [x for x in row if x]
            merged, sc, i = [], 0, 0
            while i < len(nz):
                if i+1<len(nz) and nz[i]==nz[i+1]:
                    merged.append(nz[i]+1)
                    sc += (1 << (nz[i]+1))
                    i += 2
                else:
                    merged.append(nz[i]); i += 1
            res = merged + [0]*(4-len(merged))
            return res, sc

        for i in range(65536):
            row = int_to_row(i)
            left_r,  sc_l = move_row_left(row)
            right_r, sc_r = move_row_left(row[::-1])
            right_r = right_r[::-1]

            cls._left_table[i]  = row_to_int(left_r)
            cls._right_table[i] = row_to_int(right_r)
            cls._score_table[i] = (sc_l, sc_r)

            # 转置：把 4 格变成 64 位列
            c0,c1,c2,c3 = ( (i>> (4*k)) & 0xF for k in range(4) )
            cls._transpose_tbl[i] = (
                (c0 << (4*0)) |
                (c1 << (4*4)) |
                (c2 << (4*8)) |
                (c3 << (4*12))
            )

    @classmethod
    def reset_board(cls) -> int:
        """返回一个新对局的初始棋盘整数（随机两块）。"""
        cls._init_tables()
        board = 0
        board = cls.add_random_tile(board)
        board = cls.add_random_tile(board)
        return board

    @classmethod
    def add_random_tile(cls, board:int) -> int:
        """在 board 的空格中随机放置一个 2（90%）或 4（10%），返回新的 board。"""
        empties = [i for i in range(16) if ((board >> (4*i)) & 0xF)==0]
        if not empties:
            return board
        pos = random.choice(empties)
        val = 1 if random.random()<0.9 else 2
        return board | (val << (4*pos))

    @classmethod
    def move_left(cls, board:int) -> Tuple[int,int,bool]:
        """向左移动：返回 (new_board, gained_score, moved_flag)."""
        cls._init_tables()
        new_b, total_sc, moved = 0, 0, False
        for r in range(4):
            row_int = (board >> (16*r)) & 0xFFFF
            nr = cls._left_table[row_int]
            sc = cls._score_table[row_int][0]
            if nr != row_int:
                moved = True
            new_b |= nr << (16*r)
            total_sc += sc
        return new_b, total_sc, moved

    @classmethod
    def move_right(cls, board:int) -> Tuple[int,int,bool]:
        """向右移动：返回 (new_board, gained_score, moved_flag)."""
        cls._init_tables()
        new_b, total_sc, moved = 0, 0, False
        for r in range(4):
            row_int = (board >> (16*r)) & 0xFFFF
            nr = cls._right_table[row_int]
            sc = cls._score_table[row_int][1]
            if nr != row_int:
                moved = True
            new_b |= nr << (16*r)
            total_sc += sc
        return new_b, total_sc, moved

    @classmethod
    def _transpose(cls, board:int) -> int:
        """内部：按列读取再写回，实现转置。"""
        # 将每行当列读入，再拼成新的 64 位
        out = 0
        for r in range(4):
            row_int = (board >> (16*r)) & 0xFFFF
            col_bits = cls._transpose_tbl[row_int]
            out |= col_bits << (4*r)
        return out

    @classmethod
    def move_up(cls, board:int) -> Tuple[int,int,bool]:
        """向上移动：等价 转置→左移→转置"""
        cls._init_tables()
        trans = cls._transpose(board)
        new_t, total_sc, moved = 0, 0, False
        for r in range(4):
            row_int = (trans >> (16*r)) & 0xFFFF
            nr = cls._left_table[row_int]
            sc = cls._score_table[row_int][0]
            if nr != row_int:
                moved = True
            new_t |= nr << (16*r)
            total_sc += sc
        new_board = cls._transpose(new_t)
        return new_board, total_sc, moved

    @classmethod
    def move_down(cls, board:int) -> Tuple[int,int,bool]:
        """向下移动：等价 转置→右移→转置"""
        cls._init_tables()
        trans = cls._transpose(board)
        new_t, total_sc, moved = 0, 0, False
        for r in range(4):
            row_int = (trans >> (16*r)) & 0xFFFF
            nr = cls._right_table[row_int]
            sc = cls._score_table[row_int][1]
            if nr != row_int:
                moved = True
            new_t |= nr << (16*r)
            total_sc += sc
        new_board = cls._transpose(new_t)
        return new_board, total_sc, moved

    @staticmethod
    def get_board_array(board:int) -> np.ndarray:
        """将 board→4×4 numpy 阵列，值恢复为真实 tile。"""
        arr = np.zeros((4,4),dtype=int)
        for i in range(16):
            v = (board >> (4*i)) & 0xF
            arr[i//4, i%4] = 0 if v==0 else (1<<v)
        return arr

    @classmethod
    def is_game_over(cls, board:int) -> bool:
        """检查无空格且无法合并时游戏结束。"""
        cls._init_tables()
        # 空格检查
        for i in range(16):
            if ((board>>(4*i)) & 0xF) == 0:
                return False
        # 横向可合并？
        for r in range(4):
            row_int = (board >> (16*r)) & 0xFFFF
            if cls._left_table[row_int] != row_int:
                return False
        # 纵向可合并？
        tb = cls._transpose(board)
        for r in range(4):
            row_int = (tb >> (16*r)) & 0xFFFF
            if cls._left_table[row_int] != row_int:
                return False
        return True

    @staticmethod
    def get_max_tile(board:int) -> int:
        """返回当前最大 tile（2^n）的真实值。"""
        m = 0
        tmp = board
        for _ in range(16):
            m = max(m, tmp & 0xF)
            tmp >>= 4
        return 0 if m==0 else (1<<m)

class TestGame2048(unittest.TestCase):
    def setUp(self):
        # 固定随机种子，以便 reset_board 的 tile 位置可复现
        random.seed(42)

    def test_board_initialization(self):
        board = Game2048.reset_board()
        arr = Game2048.get_board_array(board)
        # 应该恰好有两个非零格
        self.assertEqual(np.count_nonzero(arr), 2)

    def test_move_left_simple(self):
        # 在 (0,0) 和 (0,2) 各放一个 “2”
        board = (1 << (4*0)) | (1 << (4*2))
        new_b, sc, moved = Game2048.move_left(board)
        # 应该合并成一个 “4” 在 (0,0)
        self.assertTrue(moved)
        self.assertEqual((new_b >> (4*0)) & 0xF, 2)  # nibble=2 -> tile=4
        self.assertEqual(sc, 1 << (1+1))             # 得分 2^(1+1)=4

    def test_move_right_simple(self):
        board = (1 << (4*0)) | (1 << (4*2))
        new_b, sc, moved = Game2048.move_right(board)
        # 应该合并成一个 “4” 在 (0,3)
        self.assertTrue(moved)
        self.assertEqual((new_b >> (4*3)) & 0xF, 2)
        self.assertEqual(sc, 1 << (1+1))

    def test_transpose(self):
        # 构造一行 [2,4,8,16] -> nibble [1,2,3,4]
        board = (
            (1 << (4*0)) |
            (2 << (4*1)) |
            (3 << (4*2)) |
            (4 << (4*3))
        )
        t = Game2048._transpose(board)
        # 期望变成一列 [2,4,8,16]（其余为0）
        expected = (
            (1 << (4*0)) |
            (2 << (4*4)) |
            (3 << (4*8)) |
            (4 << (4*12))
        )
        self.assertEqual(t, expected)

    def test_move_up_down(self):
        # 在 (0,0) 和 (2,0) 各放一个 “2”
        board = (1 << (4*0)) | (1 << (4*8))
        # 向上
        up_b, up_sc, up_mv = Game2048.move_up(board)
        self.assertTrue(up_mv)
        # 合并到 (0,0)
        self.assertEqual((up_b >> (4*0)) & 0xF, 2)
        self.assertEqual(up_sc, 1 << (1+1))
        # 向下
        dn_b, dn_sc, dn_mv = Game2048.move_down(board)
        self.assertTrue(dn_mv)
        # 合并到 (3,0)
        self.assertEqual((dn_b >> (4*12)) & 0xF, 2)
        self.assertEqual(dn_sc, 1 << (1+1))

    def test_game_over_detection(self):
        # 填满棋盘且无可合并格
        vals = [1,2,1,2,2,1,2,1,1,2,1,2,2,1,2,1]
        board = 0
        for i, v in enumerate(vals):
            board |= (v << (4*i))
        self.assertTrue(Game2048.is_game_over(board))

    def test_max_tile(self):
        board = 3 << (4*0)  # 唯一的方块是 8
        self.assertEqual(Game2048.get_max_tile(board), 8)

    def test_lookup_via_move(self):
        # 间接验证 left_table/right_table
        board = (1 << (4*0)) | (1 << (4*1))  # [2,2,0,0]
        lb, sl, ml = Game2048.move_left(board)
        self.assertEqual(lb, 2)      # 合并到 nibble=2
        self.assertEqual(sl, 4)      # 得分 4
        self.assertTrue(ml)

        rb, sr, mr = Game2048.move_right(board)
        self.assertEqual(rb, 2 << (4*3))
        self.assertEqual(sr, 4)
        self.assertTrue(mr)

if __name__ == "__main__":
    unittest.main()
