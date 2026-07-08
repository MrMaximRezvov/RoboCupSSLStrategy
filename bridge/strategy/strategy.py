"""High-level strategy code"""

import math
from typing import Optional
from bridge.const import Color
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


class Strategy:
    """Main class of strategy"""

    def __init__(self) -> None:
        self.we_active = False
        self.prev_ball = aux.Point(0, 0)
        self.active_robots = []
        self.goalkeeper_id = None
        self.attackers = []
        self.goalkeeper_arc_radius = 350.0  # Дуга вратаря перед воротами
        self.goalkeeper_intercept_offset = 200.0  # Смещение на перехват
        # Фильтр для плавности вратаря
        self.gk_target_smooth = aux.Point(0, 0)
        self.gk_smooth_alpha = 1.0  # Коэффициент сглаживания (0..1)

    def get_robots_id(self, field: fld.Field) -> list[int]:
        """Get list of active robot IDs"""
        active_robots = []
        for i in range(16):
            if field.allies[i].is_used():
                active_robots.append(i)
        return sorted(active_robots)

    def draw_robot_arrow(self, field: fld.Field, robot_pos: aux.Point, robot_angle: float):
        """Draw robot direction arrow"""
        arrow_length = 150
        end_look_x = robot_pos.x + arrow_length * math.cos(robot_angle)
        end_look_y = robot_pos.y + arrow_length * math.sin(robot_angle)

        left_ang = robot_angle + math.pi / 2
        right_ang = robot_angle - math.pi / 2

        field.strategy_image.draw_poly([
            aux.Point(robot_pos.x + 20 * math.cos(left_ang), robot_pos.y + 20 * math.sin(left_ang)),
            aux.Point(end_look_x + 20 * math.cos(left_ang), end_look_y + 20 * math.sin(left_ang)),
            aux.Point(end_look_x + 50 * math.cos(left_ang), end_look_y + 50 * math.sin(left_ang)),
            aux.Point(robot_pos.x + 200 * math.cos(robot_angle), robot_pos.y + 200 * math.sin(robot_angle)),
            aux.Point(end_look_x + 50 * math.cos(right_ang), end_look_y + 50 * math.sin(right_ang)),
            aux.Point(end_look_x + 20 * math.cos(right_ang), end_look_y + 20 * math.sin(right_ang)),
            aux.Point(robot_pos.x + 20 * math.cos(right_ang), robot_pos.y + 20 * math.sin(right_ang)),
        ])

    def draw_all_robots(self, field: fld.Field):
        """Draw all robots on the field"""
        for i in range(len(field.allies)):
            robot_pos = field.allies[i].get_pos()
            robot_angle = field.allies[i].get_angle()
            self.draw_robot_arrow(field, robot_pos, robot_angle)

        for i in range(len(field.enemies)):
            robot_pos = field.enemies[i].get_pos()
            robot_angle = field.enemies[i].get_angle()
            self.draw_robot_arrow(field, robot_pos, robot_angle)

    def get_goalkeeper_target(self, field: fld.Field) -> aux.Point:
        """
        Calculate goalkeeper target position.
        Логика:
        1. Вратарь всегда стоит на дуге перед воротами (не на линии!)
        2. Если мяч летит в створ - выезжает на перехват по траектории
        3. Если мяч за воротами/полем - стоит в центре
        4. Плавное сглаживание позиции, чтобы не дёргался
        """
        curr_ball = field.ball.get_pos()
        our_goal = field.ally_goal

        max_y = max(our_goal.up.y, our_goal.down.y)
        min_y = min(our_goal.up.y, our_goal.down.y)
        goal_x = our_goal.center.x


        direction_sign = 1 if our_goal.eye_forw.x > 0 else -1

        ball_behind_goal_line = (
            (direction_sign > 0 and curr_ball.x < goal_x) or
            (direction_sign < 0 and curr_ball.x > goal_x)
        )

        ball_out_of_field = (
            abs(curr_ball.x) > const.FIELD_DX / 2 or
            abs(curr_ball.y) > const.FIELD_DY / 2
        )

        gk_line_x = goal_x + direction_sign * self.goalkeeper_arc_radius

        if ball_behind_goal_line or ball_out_of_field:
            target = aux.Point(gk_line_x, our_goal.center.y)
            field.strategy_image.draw_circle(target, (100, 100, 100), 50)
            return self._smooth_gk_target(target)


        ball_vel = field.ball.get_vel()
        ball_speed = ball_vel.mag()

        ball_approaching_our_goal = (
            (direction_sign > 0 and ball_vel.x < -50) or  
            (direction_sign < 0 and ball_vel.x > 50) 
        )

        if ball_approaching_our_goal and ball_speed > 100:
            if abs(ball_vel.x) > 1e-2:
                t = (goal_x - curr_ball.x) / ball_vel.x
                if t > 0:  
                    predicted_y = curr_ball.y + ball_vel.y * t

                    goal_margin = 150
                    if (min_y - goal_margin) <= predicted_y <= (max_y + goal_margin):
                        intercept_x = goal_x + direction_sign * self.goalkeeper_intercept_offset
                        final_y = max(min_y, min(max_y, predicted_y))
                        target = aux.Point(intercept_x, final_y)

                        field.strategy_image.draw_line(
                            curr_ball,
                            aux.Point(goal_x, predicted_y),
                            (255, 0, 0)
                        )
                        field.strategy_image.draw_circle(target, (255, 0, 0), 60)
                        return self._smooth_gk_target(target)

        vec_to_up = our_goal.up - curr_ball
        vec_to_down = our_goal.down - curr_ball

        if vec_to_up.mag() < 1e-5 or vec_to_down.mag() < 1e-5:
            target = aux.Point(gk_line_x, our_goal.center.y)
            return self._smooth_gk_target(target)

        bisector_dir = (vec_to_up.unity() + vec_to_down.unity())
        if bisector_dir.mag() < 1e-5:
            target = aux.Point(gk_line_x, our_goal.center.y)
            return self._smooth_gk_target(target)
        bisector_dir = bisector_dir.unity()

        if abs(bisector_dir.x) > 1e-5:
            t = (gk_line_x - curr_ball.x) / bisector_dir.x
            gk_y = curr_ball.y + bisector_dir.y * t
        else:
            gk_y = our_goal.center.y

        final_y = max(min_y - 100, min(max_y + 100, gk_y))
        target = aux.Point(gk_line_x, final_y)

        field.strategy_image.draw_line(curr_ball, our_goal.up, (0, 255, 0))
        field.strategy_image.draw_line(curr_ball, our_goal.down, (0, 255, 0))
        field.strategy_image.draw_circle(target, (0, 255, 255), 50)

        return self._smooth_gk_target(target)

    def _smooth_gk_target(self, target: aux.Point) -> aux.Point:
        """Плавное сглаживание позиции вратаря, чтобы не дёргался"""
        if self.gk_target_smooth.x == 0 and self.gk_target_smooth.y == 0:
            self.gk_target_smooth = target
            return target

        smoothed_x = self.gk_smooth_alpha * target.x + (1 - self.gk_smooth_alpha) * self.gk_target_smooth.x
        smoothed_y = self.gk_smooth_alpha * target.y + (1 - self.gk_smooth_alpha) * self.gk_target_smooth.y
        self.gk_target_smooth = aux.Point(smoothed_x, smoothed_y)
        return self.gk_target_smooth

    def get_best_shot_target(self, field: fld.Field) -> aux.Point:
        """
        Find the best target to shoot at enemy goal, avoiding their goalkeeper.
        Строим касательные от мяча к вратарю противника, находим их пересечение
        с линией ворот, выбираем точку подальше от вратаря.
        """
        curr_ball = field.ball.get_pos()
        enemy_goal = field.enemy_goal
        enemy_gk_id = field.enemy_gk_id

        goal_up = enemy_goal.up
        goal_down = enemy_goal.down
        goal_min_y = min(goal_up.y, goal_down.y)
        goal_max_y = max(goal_up.y, goal_down.y)
        goal_x = enemy_goal.center.x

        if enemy_gk_id is None or not field.enemies[enemy_gk_id].is_used():
            if curr_ball.x < goal_x:
                return goal_up if curr_ball.y < 0 else goal_down
            else:
                return goal_up if curr_ball.y > 0 else goal_down

        enemy_gk_pos = field.enemies[enemy_gk_id].get_pos()
        gk_radius = 120  

        mid_point = (curr_ball + enemy_gk_pos) / 2
        tangents_to_gk = get_tangent_points(enemy_gk_pos, mid_point, gk_radius)

        if len(tangents_to_gk) != 2:

            return enemy_goal.center

        goal_line_start = enemy_goal.up
        goal_line_end = enemy_goal.down

        valid_targets = []

        for tangent_point in tangents_to_gk:
            ray_direction = (tangent_point - curr_ball).unity()
            ray_end = curr_ball + ray_direction * 10000

            intersection = get_line_intersection(
                curr_ball,
                ray_end,
                goal_line_start,
                goal_line_end,
                "SS"  
            )

            if intersection is not None:

                if goal_min_y <= intersection.y <= goal_max_y:
                    dot = (intersection - curr_ball).x * (goal_x - curr_ball.x)
                    if dot > 0:  
                        valid_targets.append(intersection)

            field.strategy_image.draw_line(curr_ball, tangent_point, (255, 100, 0))

        field.strategy_image.draw_circle(enemy_gk_pos, (255, 0, 255), gk_radius)

        if not valid_targets:
            if enemy_gk_pos.y > enemy_goal.center.y:
                target = enemy_goal.down
            else:
                target = enemy_goal.up
            field.strategy_image.draw_circle(target, (255, 255, 0), 80)
            field.strategy_image.draw_line(curr_ball, target, (255, 255, 0))
            return target

        best_target = max(valid_targets, key=lambda p: (p - enemy_gk_pos).mag())

        field.strategy_image.draw_circle(best_target, (255, 255, 0), 80)
        field.strategy_image.draw_line(curr_ball, best_target, (255, 255, 0))

        for t in valid_targets:
            field.strategy_image.draw_circle(t, (0, 200, 200), 40)

        return best_target

    def get_approach_point(self, ball_pos: aux.Point, target_pos: aux.Point, distance: float = 180) -> aux.Point:
        """
        Calculate approach point behind the ball relative to target.
        Робот должен подъехать к мячу СЗАДИ (со стороны, противоположной цели),
        чтобы ударить мяч в цель.
        """
        direction = (target_pos - ball_pos).unity()
        approach_point = ball_pos - direction * distance
        return approach_point

    def process(self, field: fld.Field) -> list[Optional[Action]]:
        """Game State Management"""
        if field.game_state not in [GameStates.KICKOFF, GameStates.PENALTY]:
            if field.active_team in [const.Color.ALL, field.ally_color]:
                self.we_active = True
            else:
                self.we_active = False

        actions: list[Optional[Action]] = [None] * const.TEAM_ROBOTS_MAX_COUNT

        match field.game_state:
            case GameStates.RUN:
                self.run(field, actions)
            case GameStates.TIMEOUT:
                pass
            case GameStates.HALT:
                return [None] * const.TEAM_ROBOTS_MAX_COUNT
            case GameStates.PREPARE_PENALTY:
                pass
            case GameStates.PENALTY:
                pass
            case GameStates.PREPARE_KICKOFF:
                pass
            case GameStates.KICKOFF:
                pass
            case GameStates.FREE_KICK:
                pass
            case GameStates.STOP:
                self.run(field, actions)
            case GameStates.BALL_PLACEMENT:
                pass
            case GameStates.DEBUG:
                pass

        return actions

    def run(self, field: fld.Field, actions: list[Optional[Action]]) -> None:
        """One iteration of strategy"""
        curr_ball = field.ball.get_pos()

        self.active_robots = self.get_robots_id(field)
        if self.active_robots:
            self.goalkeeper_id = self.active_robots[0]
            self.attackers = self.active_robots[1:]

        self.draw_all_robots(field)

        if self.goalkeeper_id is not None:
            gk_target = self.get_goalkeeper_target(field)

            angle_to_ball = (curr_ball - gk_target).arg()
            actions[self.goalkeeper_id] = Actions.GoToPointIgnore(gk_target, angle_to_ball)

        if self.attackers:
            attacker_id = self.attackers[0]
            shot_target = self.get_best_shot_target(field)

            attacker_pos = field.allies[attacker_id].get_pos()
            attacker_angle = field.allies[attacker_id].get_angle()
            dist_to_ball = (attacker_pos - curr_ball).mag()

            kick_angle = (shot_target - curr_ball).arg()

            approach_point = self.get_approach_point(curr_ball, shot_target, distance=180)

            field.strategy_image.draw_circle(approach_point, (0, 255, 0), 40)
            field.strategy_image.draw_line(approach_point, curr_ball, (0, 255, 0))
            field.strategy_image.draw_circle(shot_target, (255, 255, 0), 60)

            robot_to_ball = curr_ball - attacker_pos

            ball_to_target = shot_target - curr_ball


            if robot_to_ball.mag() > 1e-5 and ball_to_target.mag() > 1e-5:
                r2b_dir = robot_to_ball.unity()
                b2t_dir = ball_to_target.unity()
                dot_product = r2b_dir.x * b2t_dir.x + r2b_dir.y * b2t_dir.y

                angle_diff = abs(aux.wind_down_angle(attacker_angle - kick_angle))

                is_good_position = (
                    dist_to_ball < 200 and
                    dot_product < -0.5 and
                    angle_diff < 0.4
                )

                if is_good_position:
                    actions[attacker_id] = KickActions.Straight(shot_target)
                    field.strategy_image.draw_line(curr_ball, shot_target, (255, 0, 255))
                else:
                    # actions[attacker_id] = Actions.GoToPoint(approach_point, kick_angle)
                    actions[attacker_id] = KickActions.Straight(shot_target)
            else:
                if dist_to_ball < 150:
                    actions[attacker_id] = KickActions.Straight(shot_target)
                else:
                    actions[attacker_id] = KickActions.Straight(shot_target)
                    # actions[attacker_id] = Actions.GoToPoint(approach_point, kick_angle)

        self.prev_ball = curr_ball