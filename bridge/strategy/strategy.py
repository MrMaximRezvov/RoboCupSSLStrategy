"""High-level strategy code — SSL 3v3, with proper defense"""

import math
from typing import Optional, Union

from bridge import const
from bridge.auxiliary import aux, fld, rbt
from bridge.auxiliary.aux import get_line_intersection, get_tangent_points
from bridge.const import State as GameStates
from bridge.router.actions import (
    Action,
    Actions,
    KickActions,
    StrategyActions,
)


# Константы
GK_ARC_RADIUS = 450.0
GK_FORWARD_OFFSET = 150.0
DEF_BLOCK_DISTANCE = 700.0
APPROACH_OFFSET = 500.0
GK_SHADOW_RADIUS = 120.0
MAX_KICK_VOLTAGE = 15.0
PASS_VOLTAGE = 5.0
CLEARANCE_VOLTAGE = 15.0
CONFLICT_THRESHOLD = 200.0
BALL_BEHIND_THRESHOLD = 300.0
KICK_RANGE = 500.0
APPROACH_RANGE = 100.0
GK_CLEARANCE_RANGE = 300.0
STOP_DISTANCE = 500.0
PENALTY_STAND_DIST = 1500.0
GK_ON_LINE_OFFSET = 30.0


class Strategy:
    """Главный класс стратегии"""

    def __init__(self) -> None:
        self.we_active = False
        self.prev_ball = aux.Point(0, 0)
        self.last_ball_handler: Optional[int] = None

        self.goalkeeper_id: Optional[int] = None
        self.defender_id: Optional[int] = None
        self.attacker_id: Optional[int] = None

    # Роли
    def get_active_robots(self, field: fld.Field) -> list[int]:
        return sorted(i for i in range(const.TEAM_ROBOTS_MAX_COUNT)
                      if field.allies[i].is_used())

    def assign_roles(self, field: fld.Field) -> None:
        """Вратарь = робот с наименьшим id. Остальные — по близости к мячу."""
        active = self.get_active_robots(field)
        if not active:
            self.goalkeeper_id = self.defender_id = self.attacker_id = None
            return

        self.goalkeeper_id = active[0]
        rest = active[1:]

        if not rest:
            self.defender_id = self.attacker_id = None
            return

        ball = field.ball.get_pos()

        if len(rest) == 1:
            self.attacker_id = rest[0]
            self.defender_id = None
            return

        dists = [(i, (field.allies[i].get_pos() - ball).mag()) for i in rest]
        dists.sort(key=lambda x: x[1])

        closest_id, closest_dist = dists[0]
        second_id, second_dist = dists[1]

        if abs(closest_dist - second_dist) < CONFLICT_THRESHOLD:
            if self.last_ball_handler == second_id:
                self.attacker_id = second_id
                self.defender_id = closest_id
            else:
                self.attacker_id = closest_id
                self.defender_id = second_id
        else:
            self.attacker_id = closest_id
            self.defender_id = second_id

    # Геометрия
    def is_in_ally_penalty_area(self, field: fld.Field, p: aux.Point) -> bool:
        if field.ally_goal.hull is None:
            return False
        return self._point_in_poly(p, field.ally_goal.hull)

    def is_in_enemy_penalty_area(self, field: fld.Field, p: aux.Point) -> bool:
        if field.enemy_goal.hull is None:
            return False
        return self._point_in_poly(p, field.enemy_goal.hull)

    @staticmethod
    def _point_in_poly(p: aux.Point, hull: list[aux.Point]) -> bool:
        inside = False
        n = len(hull)
        j = n - 1
        for i in range(n):
            xi, yi = hull[i].x, hull[i].y
            xj, yj = hull[j].x, hull[j].y
            if ((yi > p.y) != (yj > p.y)) and \
               (p.x < (xj - xi) * (p.y - yi) / (yj - yi + 1e-9) + xi):
                inside = not inside
            j = i
        return inside

    def _avoid_ally_penalty(self, field: fld.Field, target: aux.Point) -> aux.Point:
        if not self.is_in_ally_penalty_area(field, target):
            return target

        center = field.ally_goal.center
        diff = target - center
        if diff.mag() < 1e-3:
            ball = field.ball.get_pos()
            diff = ball - center
            if diff.mag() < 1e-3:
                diff = aux.Point(1.0, 0.0)
        dir_out = diff.unity()

        lo, hi = 0.0, diff.mag() + 1000.0
        for _ in range(25):
            mid = (lo + hi) / 2.0
            p = center + dir_out * mid
            if self.is_in_ally_penalty_area(field, p):
                lo = mid
            else:
                hi = mid
        return center + dir_out * (hi + 60.0)

    def _enforce_min_ball_distance(self, field: fld.Field, target: aux.Point,
                                    min_dist: float) -> aux.Point:
        """Сдвигает точку так, чтобы она была не ближе min_dist к мячу."""
        if min_dist <= 0:
            return target
        ball = field.ball.get_pos()
        diff = target - ball
        d = diff.mag()
        if d < min_dist:
            if d < 1e-3:
                # Если цель прямо на мяче — отъезжаем к своим воротам
                dir_out = aux.Point(field.polarity * -1.0, 0.0)
            else:
                dir_out = diff.unity()
            return ball + dir_out * (min_dist + 50.0)
        return target

    def _get_retreat_position(self, field: fld.Field, robot_id: int) -> aux.Point:
        """Позиция отступления: за мячом (между мячом и своими воротами)."""
        ball = field.ball.get_pos()
        robot_pos = field.allies[robot_id].get_pos()
        our_goal = field.ally_goal.center

        to_goal = our_goal - ball
        if to_goal.mag() < 1e-3:
            to_goal = aux.Point(field.polarity * -1.0, 0.0)

        retreat_point = ball + to_goal.unity() * (STOP_DISTANCE + 100)

        if (robot_pos - ball).mag() > STOP_DISTANCE + 200:
            return robot_pos

        return retreat_point

    def _get_penalty_stand_position(self, field: fld.Field, offset_idx: int = 0,
                                     polarity: bool = False) -> aux.Point:
        """Позиция для стоящих в PENALTY: своя/чужая половина."""
        if polarity:
            our_side_x = field.polarity * PENALTY_STAND_DIST
        else:
            our_side_x = -field.polarity * PENALTY_STAND_DIST

        perp_offset = (offset_idx - 0.5) * 600
        return aux.Point(our_side_x, perp_offset)

    def _get_penalty_kicker_position(self, field: fld.Field) -> aux.Point:
        """Позиция для бьющего в PENALTY: за мячом, со стороны своих ворот."""
        ball = field.ball.get_pos()
        enemy_goal = field.enemy_goal.center

        to_enemy = enemy_goal - ball
        if to_enemy.mag() < 1e-3:
            to_enemy = aux.Point(field.polarity * 1.0, 0.0)

        approach = ball - to_enemy.unity() * APPROACH_OFFSET
        return approach

    def _get_defense_block_position(self, field: fld.Field, offset_side: float = 0.0) -> aux.Point:
        """
        Позиция для блокировки линии мяч-наши ворота.
        offset_side — боковое смещение (для расстановки нескольких роботов).
        """
        ball = field.ball.get_pos()
        our_goal = field.ally_goal.center

        to_goal = our_goal - ball
        if to_goal.mag() < 1e-3:
            return our_goal

        dir_to_goal = to_goal.unity()
        # Перпендикуляр для бокового смещения
        perp = aux.Point(-dir_to_goal.y, dir_to_goal.x)

        block_point = ball + dir_to_goal * DEF_BLOCK_DISTANCE + perp * offset_side
        return block_point

    # ВРАТАРЬ
    def get_goalkeeper_action(self, field: fld.Field) -> Union[aux.Point, Action]:
        curr_ball = field.ball.get_pos()
        gk_pos = field.allies[self.goalkeeper_id].get_pos()
        gk_angle = field.allies[self.goalkeeper_id].get_angle()

        if self.is_in_ally_penalty_area(field, curr_ball):
            pass_target = field.enemy_goal.center
            kick_angle = (pass_target - curr_ball).arg()
            angle_to_ball = (curr_ball - gk_pos).arg()
            dist_to_ball = (gk_pos - curr_ball).mag()
            angle_diff = abs(aux.wind_down_angle(gk_angle - kick_angle))

            field.strategy_image.draw_line(curr_ball, pass_target, (0, 255, 255))
            field.strategy_image.draw_circle(curr_ball, (255, 0, 255), GK_CLEARANCE_RANGE)

            if dist_to_ball < GK_CLEARANCE_RANGE and angle_diff < 0.5:
                return KickActions.Straight(pass_target, voltage=CLEARANCE_VOLTAGE, is_upper=True)

            if dist_to_ball < BALL_BEHIND_THRESHOLD:
                angle_diff_to_ball = abs(aux.wind_down_angle(gk_angle - angle_to_ball))
                if angle_diff_to_ball > math.pi / 2:
                    return Actions.BallGrab(angle_to_ball)

            return Actions.GoToPoint(curr_ball, kick_angle, ball_catch=True)

        goal = field.ally_goal
        goal_x = goal.center.x
        min_y = min(goal.up.y, goal.down.y)
        max_y = max(goal.up.y, goal.down.y)

        v_up = (goal.up - curr_ball).unity()
        v_down = (goal.down - curr_ball).unity()
        bisector = (v_up + v_down).unity()

        if field.ally_color == const.Color.BLUE:
            direction_to_field = bisector * (-goal.eye_forw.x)
        else:
            direction_to_field = bisector * (goal.eye_forw.x)

        arc_point = goal.center + direction_to_field * GK_ARC_RADIUS

        dy = curr_ball.y - self.prev_ball.y
        dx = curr_ball.x - self.prev_ball.x

        flying_to_goal = (abs(dx) > 1e-2 and
                          ((goal_x > curr_ball.x and dx > 0) or
                           (goal_x < curr_ball.x and dx < 0)))

        if flying_to_goal:
            intercept_x = goal_x + goal.eye_forw.x * GK_FORWARD_OFFSET
            predicted_y = curr_ball.y + (intercept_x - curr_ball.x) * dy / dx
            final_y = max(min_y, min(max_y, predicted_y))
            intercept_point = aux.Point(intercept_x, final_y)
            field.strategy_image.draw_line(curr_ball, intercept_point, (255, 0, 0))
        else:
            final_y = max(min_y, min(max_y, arc_point.y))
            intercept_point = aux.Point(arc_point.x, final_y)
            field.strategy_image.draw_line(curr_ball, goal.up, (0, 255, 0))
            field.strategy_image.draw_line(curr_ball, goal.down, (0, 255, 0))
            field.strategy_image.draw_line(goal.center, intercept_point, (0, 255, 255))

        field.strategy_image.draw_circle(intercept_point, (0, 255, 255), 50)
        return intercept_point

    def get_goalkeeper_on_goal_line(self, field: fld.Field) -> aux.Point:
        """Вратарь на линии ворот, следит за проекцией мяча."""
        ball = field.ball.get_pos()
        goal = field.ally_goal

        goal_line_x = goal.center.x + goal.eye_forw.x * GK_ON_LINE_OFFSET
        min_y = min(goal.up.y, goal.down.y)
        max_y = max(goal.up.y, goal.down.y)

        target_y = max(min_y, min(max_y, ball.y))

        target = aux.Point(goal_line_x, target_y)
        field.strategy_image.draw_circle(target, (255, 0, 255), 60)
        field.strategy_image.draw_line(goal.up, goal.down, (255, 0, 255))
        return target

    def get_goalkeeper_penalty_action(self, field: fld.Field) -> Action:
        """Поведение вратаря в PENALTY: вынос если поймал, иначе на линии."""
        ball = field.ball.get_pos()
        gk_pos = field.allies[self.goalkeeper_id].get_pos()
        gk_angle = field.allies[self.goalkeeper_id].get_angle()
        enemy_goal = field.enemy_goal.center

        if self.is_in_ally_penalty_area(field, ball):
            pass_target = enemy_goal
            kick_angle = (pass_target - ball).arg()
            dist_to_ball = (gk_pos - ball).mag()
            angle_diff = abs(aux.wind_down_angle(gk_angle - kick_angle))

            field.strategy_image.draw_line(ball, pass_target, (255, 0, 255))

            if dist_to_ball < GK_CLEARANCE_RANGE and angle_diff < 0.5:
                return KickActions.Straight(pass_target, voltage=CLEARANCE_VOLTAGE)

            if dist_to_ball < BALL_BEHIND_THRESHOLD:
                angle_to_ball = (ball - gk_pos).arg()
                angle_diff_to_ball = abs(aux.wind_down_angle(gk_angle - angle_to_ball))
                if angle_diff_to_ball > math.pi / 2:
                    return Actions.BallGrab(angle_to_ball)

            return Actions.GoToPoint(ball, kick_angle, ball_catch=True)

        gk_target = self.get_goalkeeper_on_goal_line(field)
        angle = (ball - gk_target).arg()
        return Actions.GoToPointIgnore(gk_target, angle)

    # ЗАЩИТНИК
    def get_defender_target(self, field: fld.Field) -> aux.Point:
        ball = field.ball.get_pos()
        our_goal = field.ally_goal.center

        to_our_goal = our_goal - ball
        dist_ball_to_goal = to_our_goal.mag()

        if dist_ball_to_goal > 1e-3:
            block_dir = to_our_goal.unity()
            block_point = ball + block_dir * DEF_BLOCK_DISTANCE
        else:
            block_point = (ball + our_goal) / 2

        field.strategy_image.draw_circle(block_point, (0, 150, 255), 40)
        field.strategy_image.draw_line(ball, our_goal, (0, 150, 255))
        return block_point

    def get_defender_action(self, field: fld.Field) -> Action:
        ball = field.ball.get_pos()
        def_pos = field.allies[self.defender_id].get_pos()
        def_angle = field.allies[self.defender_id].get_angle()

        att_pos = (field.allies[self.attacker_id].get_pos()
                   if self.attacker_id is not None else None)
        dist_def_to_ball = (def_pos - ball).mag()
        dist_att_to_ball = (att_pos - ball).mag() if att_pos is not None else float('inf')

        in_defense_zone = (ball.x * field.ally_goal.center.x > 0 or
                           (ball - field.ally_goal.center).mag() < 3000)

        if in_defense_zone and dist_def_to_ball < dist_att_to_ball - CONFLICT_THRESHOLD:
            angle_to_ball = (ball - def_pos).arg()

            if dist_def_to_ball < BALL_BEHIND_THRESHOLD:
                angle_diff = abs(aux.wind_down_angle(def_angle - angle_to_ball))
                if angle_diff > math.pi / 2:
                    self.last_ball_handler = self.defender_id
                    return Actions.BallGrab(angle_to_ball)

            if self.attacker_id is not None:
                att_target = field.allies[self.attacker_id].get_pos()
                self.last_ball_handler = self.defender_id
                return KickActions.Straight(att_target, voltage=PASS_VOLTAGE)

            self.last_ball_handler = self.defender_id
            return Actions.BallGrab(angle_to_ball)

        def_target = self.get_defender_target(field)
        def_target = self._avoid_ally_penalty(field, def_target)
        angle = (ball - def_target).arg()
        return Actions.GoToPoint(def_target, angle)

    # НАПАДАЮЩИЙ
    def get_best_shot_target(self, field: fld.Field) -> aux.Point:
        ball = field.ball.get_pos()
        goal = field.enemy_goal
        goal_x = goal.center.x
        goal_min_y = min(goal.up.y, goal.down.y)
        goal_max_y = max(goal.up.y, goal.down.y)

        gk_id = field.enemy_gk_id
        if gk_id is None or not field.enemies[gk_id].is_used():
            return goal.center

        gk_pos = field.enemies[gk_id].get_pos()
        mid_point = (ball + gk_pos) / 2
        tangents = get_tangent_points(gk_pos, mid_point, GK_SHADOW_RADIUS)
        if len(tangents) != 2:
            return goal.center

        shadow_ys = []
        for tp in tangents:
            ray_end = ball + (tp - ball).unity() * 10000
            inter = get_line_intersection(ball, ray_end, goal.up, goal.down, "SS")
            if inter is not None:
                dot = (inter - ball).x * (goal_x - ball.x)
                if dot > 0:
                    shadow_ys.append(inter.y)
            field.strategy_image.draw_line(ball, tp, (255, 100, 0))

        field.strategy_image.draw_circle(gk_pos, (255, 0, 255), GK_SHADOW_RADIUS)

        if len(shadow_ys) != 2:
            return goal.center

        sh_min, sh_max = min(shadow_ys), max(shadow_ys)

        z1_w = max(0, sh_min - goal_min_y)
        z1_c = (goal_min_y + sh_min) / 2
        z2_w = max(0, goal_max_y - sh_max)
        z2_c = (sh_max + goal_max_y) / 2

        if z1_w > 10:
            field.strategy_image.draw_line(
                aux.Point(goal_x - 50, goal_min_y),
                aux.Point(goal_x - 50, sh_min), (0, 255, 0))
        if z2_w > 10:
            field.strategy_image.draw_line(
                aux.Point(goal_x - 50, sh_max),
                aux.Point(goal_x - 50, goal_max_y), (0, 255, 0))

        if z1_w < 10 and z2_w < 10:
            target_y = goal_min_y + 50 if gk_pos.y > goal.center.y else goal_max_y - 50
        elif z1_w > z2_w:
            target_y = z1_c
        else:
            target_y = z2_c

        target = aux.Point(goal_x, target_y)
        field.strategy_image.draw_circle(target, (255, 255, 0), 80)
        field.strategy_image.draw_line(ball, target, (255, 255, 0))
        return target

    def get_attacker_action(self, field: fld.Field) -> Action:
        ball = field.ball.get_pos()
        att = field.allies[self.attacker_id]
        att_pos = att.get_pos()
        att_angle = att.get_angle()

        if self.is_in_ally_penalty_area(field, ball):
            pass_target = field.enemy_goal.center
            field.strategy_image.draw_line(ball, pass_target, (0, 255, 255))
            self.last_ball_handler = self.attacker_id
            return KickActions.Straight(pass_target, voltage=PASS_VOLTAGE)

        shot_target = self.get_best_shot_target(field)
        kick_angle = (shot_target - ball).arg()

        dist_to_ball = (att_pos - ball).mag()
        angle_to_ball = (ball - att_pos).arg()

        if dist_to_ball < BALL_BEHIND_THRESHOLD:
            angle_diff = abs(aux.wind_down_angle(att_angle - angle_to_ball))
            if angle_diff > math.pi / 2:
                self.last_ball_handler = self.attacker_id
                return Actions.BallGrab(angle_to_ball)

        approach_point = ball - (shot_target - ball).unity() * APPROACH_OFFSET
        dist_to_approach = (att_pos - approach_point).mag()

        field.strategy_image.draw_circle(approach_point, (0, 255, 0), 40)
        field.strategy_image.draw_line(approach_point, ball, (0, 255, 0))

        dist_to_goal = (shot_target - ball).mag()
        voltage = MAX_KICK_VOLTAGE if dist_to_goal < 3000 else MAX_KICK_VOLTAGE * 0.7

        if dist_to_ball < KICK_RANGE or dist_to_approach < APPROACH_RANGE:
            self.last_ball_handler = self.attacker_id
            return KickActions.Straight(shot_target, voltage=voltage)

        return Actions.GoToPoint(ball, kick_angle)


    def _get_defense_positions(self, field: fld.Field, min_ball_dist: float
                                ) -> dict[int, tuple[aux.Point, float]]:
        """
        Оборонительные позиции:
          - Вратарь — на дуге перед воротами (не на линии!),
                      чтобы был выдвинут вперёд и мог свободно маневрировать.
          - Защитник и нападающий — плотно на линии мяч-ворота,
                      достаточно далеко от ворот, чтобы не мешать вратарю.
        min_ball_dist — минимальная дистанция до мяча (500 для STOP/KICKOFF, 0 для FREE_KICK).
        """
        ball = field.ball.get_pos()
        our_goal = field.ally_goal.center
        positions: dict[int, tuple[aux.Point, float]] = {}

        # Минимальное расстояние от ворот, на котором могут стоять блокеры,
        # чтобы вратарь на дуге (450мм) мог свободно двигаться между ними и воротами.
        min_blocker_dist_from_goal = GK_ARC_RADIUS + 200.0  # 650мм от ворот

        # --- Вратарь ---
        # Используем обычную логику вратаря (дуга + предсказание),
        # а не стояние на линии ворот. Это даёт вратарю пространство для манёвра.
        if self.goalkeeper_id is not None:
            gk_result = self.get_goalkeeper_action(field)
            if isinstance(gk_result, aux.Point):
                gk_target = gk_result
                # Обеспечиваем минимальную дистанцию до мяча (для STOP)
                gk_target = self._enforce_min_ball_distance(field, gk_target, min_ball_dist)
                angle = (ball - gk_target).arg()
                positions[self.goalkeeper_id] = (gk_target, angle)
            else:
                # Action (вынос/BallGrab) — в обороне не нужно, стоим на дуге
                gk_target = self.get_goalkeeper_on_goal_line(field)
                # Но выдвигаем вперёд на дугу
                goal = field.ally_goal
                gk_target = aux.Point(
                    goal.center.x + goal.eye_forw.x * GK_ARC_RADIUS * 0.5,
                    gk_target.y
                )
                gk_target = self._enforce_min_ball_distance(field, gk_target, min_ball_dist)
                angle = (ball - gk_target).arg()
                positions[self.goalkeeper_id] = (gk_target, angle)

        # --- Блокеры (защитник + нападающий) ---
        # Направление от мяча к нашим воротам
        to_goal = our_goal - ball
        if to_goal.mag() < 1e-3:
            to_goal = aux.Point(field.polarity * -1.0, 0.0)
        dir_to_goal = to_goal.unity()
        perp = aux.Point(-dir_to_goal.y, dir_to_goal.x)

        # Расстояние от мяча до линии блокеров:
        # берём максимум из DEF_BLOCK_DISTANCE и min_blocker_dist_from_goal
        dist_ball_to_goal = to_goal.mag()
        blocker_dist_from_ball = max(
            DEF_BLOCK_DISTANCE,
            dist_ball_to_goal - min_blocker_dist_from_goal
        )
        # Не ставим блокеры дальше мяча (за мячом)
        blocker_dist_from_ball = max(100.0, min(blocker_dist_from_ball, dist_ball_to_goal - 100.0))

        # Базовая точка на линии мяч-ворота
        base_block = ball + dir_to_goal * blocker_dist_from_ball

        # Расстояние между блокерами: чуть больше диаметра робота,
        # чтобы не было дыр, но и не перекрывались.
        # Диаметр робота ~180мм, ставим зазор ~100мм между краями => ~140мм от центра
        blocker_half_gap = 90.0

        # Определяем, какой робот слева, какой справа (по Y мяча)
        # Это нужно, чтобы блокеры не менялись местами каждый кадр
        defender_side = -1.0  # защитник слева
        attacker_side = 1.0   # нападающий справа

        if self.defender_id is not None:
            block = base_block + perp * (blocker_half_gap * defender_side)
            block = self._avoid_ally_penalty(field, block)
            block = self._enforce_min_ball_distance(field, block, min_ball_dist)
            positions[self.defender_id] = (block, (ball - block).arg())
            field.strategy_image.draw_circle(block, (0, 150, 255), 40)

        if self.attacker_id is not None:
            block = base_block + perp * (blocker_half_gap * attacker_side)
            block = self._avoid_ally_penalty(field, block)
            block = self._enforce_min_ball_distance(field, block, min_ball_dist)
            positions[self.attacker_id] = (block, (ball - block).arg())
            field.strategy_image.draw_circle(block, (0, 255, 150), 40)

        # Отладка: рисуем линию блокеров и позицию вратаря
        if self.defender_id is not None and self.attacker_id is not None:
            def_pos = positions[self.defender_id][0]
            att_pos = positions[self.attacker_id][0]
            field.strategy_image.draw_line(def_pos, att_pos, (0, 200, 255))
        if self.goalkeeper_id is not None:
            gk_pos = positions[self.goalkeeper_id][0]
            field.strategy_image.draw_circle(gk_pos, (255, 0, 255), 50)

        return positions
    # # ОБОРОНА (универсальная для STOP / FREE_KICK не наш / KICKOFF не наш)
    # def _get_defense_positions(self, field: fld.Field, min_ball_dist: float
    #                             ) -> dict[int, tuple[aux.Point, float]]:
    #     """
    #     Оборонительные позиции:
    #       - Вратарь — в воротах (на линии, следит за мячом).
    #       - Защитник — блокирует линию мяч-наши ворота (сбоку).
    #       - Нападающий — тоже блокирует (с другой стороны) или отъезжает.
    #     min_ball_dist — минимальная дистанция до мяча (500 для STOP/KICKOFF, 0 для FREE_KICK).
    #     """
    #     ball = field.ball.get_pos()
    #     our_goal = field.ally_goal.center
    #     positions: dict[int, tuple[aux.Point, float]] = {}

    #     # Вратарь — на линии ворот
    #     if self.goalkeeper_id is not None:
    #         gk_target = self.get_goalkeeper_on_goal_line(field)
    #         positions[self.goalkeeper_id] = (gk_target, (ball - gk_target).arg())
    #         # gk_result = self.get_goalkeeper_action(field)
    #         # if isinstance(gk_result, aux.Point):
    #             # angle_to_ball = (ball - gk_result).arg()
    #             # positions[self.goalkeeper_id] = (gk_result, angle_to_ball)
    #             # actions[self.goalkeeper_id] = Actions.GoToPoint(gk_result, angle_to_ball)
    #         # else:
    #             # actions[self.goalkeeper_id] = gk_result

    #     # Защитник — блокирует линию мяч-ворота слева
    #     if self.defender_id is not None:
    #         block = self._get_defense_block_position(field, offset_side=-150.0)
    #         block = self._avoid_ally_penalty(field, block)
    #         block = self._enforce_min_ball_distance(field, block, min_ball_dist)
    #         positions[self.defender_id] = (block, (ball - block).arg())

    #     # Нападающий — блокирует справа (или отъезжает, если далеко)
    #     if self.attacker_id is not None:
    #         block = self._get_defense_block_position(field, offset_side=150.0)
    #         block = self._avoid_ally_penalty(field, block)
    #         block = self._enforce_min_ball_distance(field, block, min_ball_dist)
    #         positions[self.attacker_id] = (block, (ball - block).arg())

    #     return positions

    # СТАНДАРТНЫЕ ПОЛОЖЕНИЯ
    def handle_stop(self, field: fld.Field, actions: list[Optional[Action]]) -> None:
        """STOP: оборона + отъезд на 500мм от мяча."""
        self.assign_roles(field)
        positions = self._get_defense_positions(field, min_ball_dist=STOP_DISTANCE)

        for rid, (target, angle) in positions.items():
            if rid == self.goalkeeper_id:
                # Вратарь игнорирует препятствия (своя штрафная)
                actions[rid] = Actions.GoToPointIgnore(target, angle)
            else:
                actions[rid] = Actions.GoToPoint(target, angle)

    def _kickoff_positions(self, field: fld.Field) -> dict[int, tuple[aux.Point, float]]:
        """Позиции для KICKOFF/PREPARE_KICKOFF (атака)."""
        ball = field.ball.get_pos()
        our_goal = field.ally_goal.center
        enemy_goal = field.enemy_goal.center
        positions = {}

        if self.goalkeeper_id is not None:
            gk_target = field.ally_goal.center
            positions[self.goalkeeper_id] = (gk_target, (ball - gk_target).arg())

        if self.attacker_id is not None:
            to_goal = our_goal - ball
            if to_goal.mag() < 1e-3:
                to_goal = aux.Point(field.polarity * -1.0, 0.0)
            approach = ball + to_goal.unity() * APPROACH_OFFSET
            positions[self.attacker_id] = (approach, (enemy_goal - ball).arg())

        if self.defender_id is not None:
            side_point = ball + aux.Point(400 * field.polarity, 400)
            positions[self.defender_id] = (side_point, (enemy_goal - side_point).arg())

        return positions

    def handle_prepare_kickoff(self, field: fld.Field, actions: list[Optional[Action]]) -> None:
        """PREPARE_KICKOFF: занимаем позиции и НЕ двигаемся."""
        self.assign_roles(field)

        if self.we_active:
            # Мы бьём — атакующие позиции
            positions = self._kickoff_positions(field)
        else:
            # Бьют они — оборона с дистанцией 500мм
            positions = self._get_defense_positions(field, min_ball_dist=STOP_DISTANCE)

        for rid, (target, angle) in positions.items():
            if rid == self.goalkeeper_id:
                actions[rid] = Actions.GoToPointIgnore(target, angle)
            else:
                actions[rid] = Actions.GoToPoint(target, angle)

    def handle_kickoff(self, field: fld.Field, actions: list[Optional[Action]], we_kick: bool) -> None:
        """KICKOFF: если мы бьём — нападающий бьёт, иначе оборона."""
        self.assign_roles(field)
        ball = field.ball.get_pos()
        enemy_goal = field.enemy_goal.center

        if we_kick:
            if self.attacker_id is not None:
                dist_to_ball = (field.allies[self.attacker_id].get_pos() - ball).mag()
                if dist_to_ball < KICK_RANGE:
                    actions[self.attacker_id] = KickActions.Straight(enemy_goal, voltage=MAX_KICK_VOLTAGE)
                else:
                    positions = self._kickoff_positions(field)
                    target, angle = positions[self.attacker_id]
                    actions[self.attacker_id] = Actions.GoToPoint(target, angle)

            positions = self._kickoff_positions(field)
            for rid in (self.goalkeeper_id, self.defender_id):
                if rid is None or rid not in positions:
                    continue
                target, angle = positions[rid]
                if rid == self.goalkeeper_id:
                    actions[rid] = Actions.GoToPointIgnore(target, angle)
                else:
                    actions[rid] = Actions.GoToPoint(target, angle)
        else:
            # Оборона с дистанцией 500мм
            positions = self._get_defense_positions(field, min_ball_dist=STOP_DISTANCE)
            for rid, (target, angle) in positions.items():
                if rid == self.goalkeeper_id:
                    actions[rid] = Actions.GoToPointIgnore(target, angle)
                else:
                    actions[rid] = Actions.GoToPoint(target, angle)

    def _penalty_positions(self, field: fld.Field, we_kick: bool) -> dict[int, tuple[aux.Point, float]]:
        """Позиции для PENALTY/PREPARE_PENALTY."""
        ball = field.ball.get_pos()
        enemy_goal = field.enemy_goal.center
        positions = {}

        if self.goalkeeper_id is not None:
            gk_target = self.get_goalkeeper_on_goal_line(field)
            look_dir = enemy_goal if we_kick else ball
            positions[self.goalkeeper_id] = (gk_target, (look_dir - gk_target).arg())

        if we_kick:
            if self.attacker_id is not None:
                att_pos = self._get_penalty_kicker_position(field)
                positions[self.attacker_id] = (att_pos, (enemy_goal - att_pos).arg())

            if self.defender_id is not None:
                def_pos = self._get_penalty_stand_position(field, offset_idx=0, polarity=True)
                positions[self.defender_id] = (def_pos, (ball - def_pos).arg())
        else:
            if self.attacker_id is not None:
                att_pos = self._get_penalty_stand_position(field, offset_idx=0)
                positions[self.attacker_id] = (att_pos, (ball - att_pos).arg())

            if self.defender_id is not None:
                def_pos = self._get_penalty_stand_position(field, offset_idx=1)
                positions[self.defender_id] = (def_pos, (ball - def_pos).arg())

        return positions

    def handle_prepare_penalty(self, field: fld.Field, actions: list[Optional[Action]]) -> None:
        """PREPARE_PENALTY: занимаем позиции и НЕ двигаемся."""
        self.assign_roles(field)
        positions = self._penalty_positions(field, we_kick=self.we_active)

        for rid, (target, angle) in positions.items():
            if rid == self.goalkeeper_id:
                actions[rid] = Actions.GoToPointIgnore(target, angle)
            else:
                actions[rid] = Actions.GoToPoint(target, angle)

    def handle_penalty(self, field: fld.Field, actions: list[Optional[Action]], we_kick: bool) -> None:
        """PENALTY: вратарь на линии/выносит, остальные стоят."""
        self.assign_roles(field)
        ball = field.ball.get_pos()
        enemy_goal = field.enemy_goal.center

        if self.goalkeeper_id is not None:
            actions[self.goalkeeper_id] = self.get_goalkeeper_penalty_action(field)

        if we_kick:
            if self.attacker_id is not None:
                att_pos = field.allies[self.attacker_id].get_pos()
                dist_to_ball = (att_pos - ball).mag()

                if dist_to_ball < KICK_RANGE:
                    actions[self.attacker_id] = KickActions.Straight(
                        enemy_goal, voltage=MAX_KICK_VOLTAGE)
                else:
                    kicker_pos = self._get_penalty_kicker_position(field)
                    actions[self.attacker_id] = Actions.GoToPoint(
                        kicker_pos, (enemy_goal - kicker_pos).arg())

            if self.defender_id is not None:
                def_pos = self._get_penalty_stand_position(field, offset_idx=0)
                actions[self.defender_id] = Actions.GoToPoint(def_pos, (ball - def_pos).arg())
        else:
            for rid, offset_idx in ((self.attacker_id, 0), (self.defender_id, 1)):
                if rid is None:
                    continue
                pos = self._get_penalty_stand_position(field, offset_idx=offset_idx)
                actions[rid] = Actions.GoToPoint(pos, (ball - pos).arg())

    def handle_free_kick(self, field: fld.Field, actions: list[Optional[Action]], we_kick: bool) -> None:
        """
        FREE_KICK:
          - Если наш — обычная игра (атака).
          - Если чужой — оборона (без минимальной дистанции, т.к. это не STOP).
        """
        self.assign_roles(field)
        if we_kick:
            self.run(field, actions)
        else:
            # Оборона: защитник блокирует, нападающий помогает, вратарь в воротах
            # Дистанция до мяча не ограничена (FREE_KICK не требует 500мм)
            positions = self._get_defense_positions(field, min_ball_dist=0)
            for rid, (target, angle) in positions.items():
                if rid == self.goalkeeper_id:
                    actions[rid] = Actions.GoToPointIgnore(target, angle)
                else:
                    actions[rid] = Actions.GoToPoint(target, angle)

    def handle_ball_placement(self, field: fld.Field, actions: list[Optional[Action]]) -> None:
        """BALL_PLACEMENT: нападающий несёт мяч, остальные отъезжают."""
        self.assign_roles(field)
        ball = field.ball.get_pos()
        target_pos = field.ball_placement_pos

        if target_pos is None:
            self.handle_stop(field, actions)
            return

        if self.attacker_id is not None:
            att_pos = field.allies[self.attacker_id].get_pos()
            dist_to_ball = (att_pos - ball).mag()

            if dist_to_ball < 200:
                angle_to_target = (target_pos - ball).arg()
                actions[self.attacker_id] = Actions.GoToPoint(target_pos, angle_to_target, ball_catch=True)
            else:
                angle_to_ball = (ball - att_pos).arg()
                actions[self.attacker_id] = Actions.BallGrab(angle_to_ball)

        for rid in (self.goalkeeper_id, self.defender_id):
            if rid is None:
                continue
            retreat = self._get_retreat_position(field, rid)
            angle = (ball - retreat).arg()
            actions[rid] = Actions.GoToPoint(retreat, angle)

    # Главный цикл
    def process(self, field: fld.Field) -> list[Optional[Action]]:
        """Game State Management"""

        print(f"State: {field.game_state}")
        print(f"GK: {self.goalkeeper_id}")
        print(f"ATT: {self.attacker_id}")
        print(f"DEF: {self.defender_id}")

        if field.game_state not in [GameStates.KICKOFF, GameStates.PENALTY]:
            self.we_active = field.active_team in [const.Color.ALL, field.ally_color]

        actions: list[Optional[Action]] = [None] * const.TEAM_ROBOTS_MAX_COUNT

        match field.game_state:
            case GameStates.RUN:
                self.run(field, actions)
            case GameStates.HALT:
                return [Actions.Stop()] * const.TEAM_ROBOTS_MAX_COUNT
            case GameStates.STOP:
                self.handle_stop(field, actions)
            case GameStates.PREPARE_KICKOFF:
                self.handle_prepare_kickoff(field, actions)
            case GameStates.KICKOFF:
                self.handle_kickoff(field, actions, we_kick=self.we_active)
            case GameStates.PREPARE_PENALTY:
                self.handle_prepare_penalty(field, actions)
            case GameStates.PENALTY:
                self.handle_penalty(field, actions, we_kick=self.we_active)
            case GameStates.FREE_KICK:
                self.handle_free_kick(field, actions, we_kick=self.we_active)
            # case GameStates.BALL_PLACEMENT:
            #     self.handle_ball_placement(field, actions)
            case GameStates.TIMEOUT | GameStates.DEBUG | GameStates.BALL_PLACEMENT:
                pass

        return actions

    def run(self, field: fld.Field, actions: list[Optional[Action]]) -> None:
        """Одна итерация обычной игры."""
        curr_ball = field.ball.get_pos()
        self.assign_roles(field)

        # ВРАТАРЬ
        if self.goalkeeper_id is not None:
            gk_result = self.get_goalkeeper_action(field)
            if isinstance(gk_result, aux.Point):
                angle_to_ball = (curr_ball - gk_result).arg()
                actions[self.goalkeeper_id] = Actions.GoToPoint(gk_result, angle_to_ball)
            else:
                actions[self.goalkeeper_id] = gk_result

        # ЗАЩИТНИК
        if self.defender_id is not None:
            actions[self.defender_id] = self.get_defender_action(field)

        # НАПАДАЮЩИЙ
        if self.attacker_id is not None:
            actions[self.attacker_id] = self.get_attacker_action(field)

        self.prev_ball = curr_ball