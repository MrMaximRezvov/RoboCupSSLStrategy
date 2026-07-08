"""High-level strategy code — SSL 3v3, fixed"""

import math
from typing import Optional

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
APPROACH_OFFSET = 180.0
GK_SHADOW_RADIUS = 120.0
MAX_KICK_VOLTAGE = 15.0
PASS_VOLTAGE = 5.0
CONFLICT_THRESHOLD = 200.0
BALL_BEHIND_THRESHOLD = 300.0
KICK_RANGE = 500.0          # Дистанция, с которой нападающий бьёт
APPROACH_RANGE = 100.0      # Если робот уже в точке захода — бьёт


class Strategy:
    """Главный класс стратегии"""

    def __init__(self) -> None:
        self.we_active = False
        self.prev_ball = aux.Point(0, 0)
        self.last_ball_handler: Optional[int] = None

        self.goalkeeper_id: Optional[int] = None
        self.defender_id: Optional[int] = None
        self.attacker_id: Optional[int] = None

    # ------------------------------------------------------------------
    # Роли
    # ------------------------------------------------------------------
    def get_active_robots(self, field: fld.Field) -> list[int]:
        return sorted(i for i in range(const.TEAM_ROBOTS_MAX_COUNT)
                      if field.allies[i].is_used())

    def assign_roles(self, field: fld.Field) -> None:
        """
        Вратарь = робот с наименьшим id (всегда active[0]).
        Остальные — по близости к мячу с гистерезисом.
        """
        active = self.get_active_robots(field)
        if not active:
            self.goalkeeper_id = self.defender_id = self.attacker_id = None
            return

        # Вратарь — всегда наименьший id
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

        # Нападающий — ближайший к мячу
        dists = [(i, (field.allies[i].get_pos() - ball).mag()) for i in rest]
        dists.sort(key=lambda x: x[1])

        closest_id, closest_dist = dists[0]
        second_id, second_dist = dists[1]

        # Гистерезис, чтобы не дёргались
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

    # ------------------------------------------------------------------
    # Геометрия
    # ------------------------------------------------------------------
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
        """
        Если целевая точка внутри нашей штрафной — сдвигаем её к ближайшей
        точке на границе штрафной (бинарный поиск по лучу из центра ворот).
        """
        if not self.is_in_ally_penalty_area(field, target):
            return target

        center = field.ally_goal.center
        diff = target - center
        if diff.mag() < 1e-3:
            # Если цель в самом центре — выходим вперёд по направлению к мячу
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
        # Добавляем небольшой запас, чтобы робот не стоял на самой границе
        return center + dir_out * (hi + 60.0)

    # ------------------------------------------------------------------
    # Вратарь
    # ------------------------------------------------------------------
    def get_goalkeeper_target(self, field: fld.Field):
        curr_ball = field.ball.get_pos()
        if self.is_in_ally_penalty_area(field, curr_ball):
            pass_target = field.enemy_goal.center
            field.strategy_image.draw_line(curr_ball, pass_target, (0, 255, 255))
            return KickActions.Straight(pass_target, voltage=PASS_VOLTAGE)

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

    # ------------------------------------------------------------------
    # Защитник
    # ------------------------------------------------------------------
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

        # Если защитник ближе к мячу и мяч в зоне защиты — подбираем
        if in_defense_zone and dist_def_to_ball < dist_att_to_ball - CONFLICT_THRESHOLD:
            # Мяч за спиной — разворот и взятие
            if dist_def_to_ball < BALL_BEHIND_THRESHOLD:
                angle_to_ball = (ball - def_pos).arg()
                angle_diff = abs(aux.wind_down_angle(def_angle - angle_to_ball))
                if angle_diff > math.pi / 2:
                    self.last_ball_handler = self.defender_id
                    return StrategyActions.CatchBall()

            # Пас нападающему
            if self.attacker_id is not None:
                att_target = field.allies[self.attacker_id].get_pos()
                self.last_ball_handler = self.defender_id
                return KickActions.Straight(att_target, voltage=PASS_VOLTAGE)

            self.last_ball_handler = self.defender_id
            return StrategyActions.CatchBall()

        # Иначе — блокируем, не заезжая в свою штрафную
        def_target = self.get_defender_target(field)
        def_target = self._avoid_ally_penalty(field, def_target)
        angle = (ball - def_target).arg()
        return Actions.GoToPoint(def_target, angle)

    # ------------------------------------------------------------------
    # Нападающий
    # ------------------------------------------------------------------
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
        """
        Нападающий:
          1. Мяч в нашей штрафной → вынос Straight.
          2. Мяч за спиной → разворот через CatchBall.
          3. Близко к мячу (< KICK_RANGE) ИЛИ уже в точке захода → удар.
          4. Иначе → едем к мячу, смотря на цель (не стоим!).
        """
        ball = field.ball.get_pos()
        att = field.allies[self.attacker_id]
        att_pos = att.get_pos()
        att_angle = att.get_angle()

        # 1. Вынос из своей штрафной
        if self.is_in_ally_penalty_area(field, ball):
            pass_target = field.enemy_goal.center
            field.strategy_image.draw_line(ball, pass_target, (0, 255, 255))
            self.last_ball_handler = self.attacker_id
            return KickActions.Straight(pass_target, voltage=PASS_VOLTAGE)

        shot_target = self.get_best_shot_target(field)
        kick_angle = (shot_target - ball).arg()

        dist_to_ball = (att_pos - ball).mag()

        # 2. Мяч за спиной — разворот
        if dist_to_ball < BALL_BEHIND_THRESHOLD:
            angle_to_ball = (ball - att_pos).arg()
            angle_diff = abs(aux.wind_down_angle(att_angle - angle_to_ball))
            if angle_diff > math.pi / 2:
                self.last_ball_handler = self.attacker_id
                return StrategyActions.CatchBall()

        # Точка захода: за мячом, с противоположной стороны от цели
        approach_point = ball - (shot_target - ball).unity() * APPROACH_OFFSET
        dist_to_approach = (att_pos - approach_point).mag()

        field.strategy_image.draw_circle(approach_point, (0, 255, 0), 40)
        field.strategy_image.draw_line(approach_point, ball, (0, 255, 0))

        # Сила удара
        dist_to_goal = (shot_target - ball).mag()
        voltage = MAX_KICK_VOLTAGE if dist_to_goal < 3000 else MAX_KICK_VOLTAGE * 0.7

        # 3. Близко к мячу ИЛИ уже в точке захода → бьём
        if dist_to_ball < KICK_RANGE or dist_to_approach < APPROACH_RANGE:
            self.last_ball_handler = self.attacker_id
            return KickActions.Straight(shot_target, voltage=voltage)

        # 4. Едем к мячу, смотря на цель (GoToPoint, а не Straight — не стоим)
        return Actions.GoToPoint(ball, kick_angle)

    # ------------------------------------------------------------------
    # Главный цикл
    # ------------------------------------------------------------------
    def process(self, field: fld.Field) -> list[Optional[Action]]:
        actions: list[Optional[Action]] = [None] * const.TEAM_ROBOTS_MAX_COUNT

        match field.game_state:
            case GameStates.RUN:
                self.run(field, actions)
            case GameStates.HALT:
                return [None] * const.TEAM_ROBOTS_MAX_COUNT
            case GameStates.STOP:
                self.run(field, actions)
            case _:
                pass

        return actions

    def run(self, field: fld.Field, actions: list[Optional[Action]]) -> None:
        curr_ball = field.ball.get_pos()
        self.assign_roles(field)

        # ВРАТАРЬ
        if self.goalkeeper_id is not None:
            gk_target = self.get_goalkeeper_target(field)
            if isinstance(gk_target, aux.Point):
                angle_to_ball = (curr_ball - gk_target).arg()
                actions[self.goalkeeper_id] = Actions.GoToPointIgnore(gk_target, angle_to_ball)
            else:
                actions[self.goalkeeper_id] = gk_target

        # ЗАЩИТНИК
        if self.defender_id is not None:
            actions[self.defender_id] = self.get_defender_action(field)

        # НАПАДАЮЩИЙ
        if self.attacker_id is not None:
            actions[self.attacker_id] = self.get_attacker_action(field)

        self.prev_ball = curr_ball