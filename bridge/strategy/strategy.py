"""High-level strategy code"""

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


class Strategy:
    """Main class of strategy"""

    def __init__(self) -> None:
        self.we_active = False
        self.prev_ball = aux.Point(0, 0)
        
        # Роли
        self.goalkeeper_id: Optional[int] = None
        self.attacker_id: Optional[int] = None

    def get_robots_id(self, field: fld.Field) -> list[int]:
        """Get list of active robot IDs"""
        active = []
        for i in range(16):
            if field.allies[i].is_used():
                active.append(i)
        return sorted(active)

    def assign_roles(self, field: fld.Field):
        """Назначает роли: первый робот - вратарь, второй - нападающий"""
        active = self.get_robots_id(field)
        self.goalkeeper_id = active[0] if len(active) > 0 else None
        self.attacker_id = active[1] if len(active) > 1 else None

    def get_goalkeeper_target(self, field: fld.Field) -> aux.Point:
        curr_ball = field.ball.get_pos()
        our_goal = field.ally_goal 
        arc_radius = 450.0

        vec_to_up = our_goal.up - curr_ball
        vec_to_down = our_goal.down - curr_ball

        bisector_dir = (vec_to_up.unity() + vec_to_down.unity()).unity()

        direction_to_field = bisector_dir * -our_goal.eye_forw.x
        arc_point = our_goal.center + direction_to_field * arc_radius

        dy = curr_ball.y - self.prev_ball.y
        dx = curr_ball.x - self.prev_ball.x
        
        goal_x = our_goal.center.x

        if abs(dx) > 1e-2 and ((goal_x > curr_ball.x and dx > 0) or (goal_x < curr_ball.x and dx < 0)):
            intercept_x = goal_x + our_goal.eye_forw.x * 150.0
            predicted_y = curr_ball.y + (intercept_x - curr_ball.x) * dy / dx

            max_y = max(our_goal.up.y, our_goal.down.y)
            min_y = min(our_goal.up.y, our_goal.down.y)
            final_y = max(min_y, min(max_y, predicted_y))
            
            intercept_point = aux.Point(intercept_x, final_y)
            field.strategy_image.draw_line(curr_ball, intercept_point, (255, 0, 0))
        else:
            max_y = max(our_goal.up.y, our_goal.down.y)
            min_y = min(our_goal.up.y, our_goal.down.y)
            final_y = max(min_y, min(max_y, arc_point.y))
            
            intercept_point = aux.Point(arc_point.x, final_y)

            field.strategy_image.draw_line(curr_ball, our_goal.up, (0, 255, 0))
            field.strategy_image.draw_line(curr_ball, our_goal.down, (0, 255, 0))
            field.strategy_image.draw_line(our_goal.center, intercept_point, (0, 255, 255))

        field.strategy_image.draw_circle(intercept_point, (0, 255, 255), 50)
        return intercept_point

    def get_best_shot_target(self, field: fld.Field) -> aux.Point:
        """Находит свободную зону в воротах через касательные к вратарю"""
        curr_ball = field.ball.get_pos()
        enemy_goal = field.enemy_goal
        enemy_gk_id = field.enemy_gk_id  

        goal_up = enemy_goal.up
        goal_down = enemy_goal.down
        goal_min_y = min(goal_up.y, goal_down.y)
        goal_max_y = max(goal_up.y, goal_down.y)
        goal_x = enemy_goal.center.x

        if enemy_gk_id is None or not field.enemies[enemy_gk_id].is_used():
            return enemy_goal.center

        enemy_gk_pos = field.enemies[enemy_gk_id].get_pos()
        gk_radius = 120

        mid_point = (curr_ball + enemy_gk_pos) / 2
        tangents = get_tangent_points(enemy_gk_pos, mid_point, gk_radius)

        if len(tangents) != 2:
            return enemy_goal.center

        shadow_ys = []
        for tp in tangents:
            ray_end = curr_ball + (tp - curr_ball).unity() * 10000
            inter = get_line_intersection(curr_ball, ray_end, goal_up, goal_down, "SS")
            if inter is not None:
                dot = (inter - curr_ball).x * (goal_x - curr_ball.x)
                if dot > 0:
                    shadow_ys.append(inter.y)
            field.strategy_image.draw_line(curr_ball, tp, (255, 100, 0))

        field.strategy_image.draw_circle(enemy_gk_pos, (255, 0, 255), gk_radius)

        if len(shadow_ys) != 2:
            return enemy_goal.center

        sh_min, sh_max = min(shadow_ys), max(shadow_ys)

        z1_w = max(0, sh_min - goal_min_y)
        z1_c = (goal_min_y + sh_min) / 2
        z2_w = max(0, goal_max_y - sh_max)
        z2_c = (sh_max + goal_max_y) / 2

        if z1_w > 10:
            field.strategy_image.draw_line(aux.Point(goal_x-50, goal_min_y), aux.Point(goal_x-50, sh_min), (0, 255, 0))
        if z2_w > 10:
            field.strategy_image.draw_line(aux.Point(goal_x-50, sh_max), aux.Point(goal_x-50, goal_max_y), (0, 255, 0))

        if z1_w < 10 and z2_w < 10:
            target_y = goal_min_y + 50 if enemy_gk_pos.y > enemy_goal.center.y else goal_max_y - 50
        elif z1_w > z2_w:
            target_y = z1_c
        else:
            target_y = z2_c

        target = aux.Point(goal_x, target_y)
        # field.strategy_image.send_telemetry()
        field.strategy_image.draw_circle(target, (255, 255, 0), 80)
        field.strategy_image.draw_line(curr_ball, target, (255, 255, 0))
        return target

    def get_attacker_action(self, field: fld.Field) -> Action:
        """Логика атакующего: правильный заход + выбор цели + удар"""
        curr_ball = field.ball.get_pos()
        attacker_pos = field.allies[self.attacker_id].get_pos()
        attacker_angle = field.allies[self.attacker_id].get_angle()

        shot_target = self.get_best_shot_target(field)
        kick_angle = (shot_target - curr_ball).arg()
        
        approach_point = curr_ball - (shot_target - curr_ball).unity() * 180
        dist_to_ball = (attacker_pos - curr_ball).mag()

        field.strategy_image.draw_circle(approach_point, (0, 255, 0), 40)
        field.strategy_image.draw_line(approach_point, curr_ball, (0, 255, 0))

        robot_to_ball = curr_ball - attacker_pos
        ball_to_target = shot_target - curr_ball

        if robot_to_ball.mag() > 1e-5 and ball_to_target.mag() > 1e-5:
            r2b_dir = robot_to_ball.unity()
            b2t_dir = ball_to_target.unity()
            dot = r2b_dir.x * b2t_dir.x + r2b_dir.y * b2t_dir.y
            angle_diff = abs(aux.wind_down_angle(attacker_angle - kick_angle))

            if dist_to_ball < 700 and dot < -0.5 and angle_diff < 0.4:
                return KickActions.Straight(shot_target, voltage=15)

        # return Actions.GoToPoint(approach_point, kick_angle)
        return KickActions.Straight(shot_target, voltage=15)


    def process(self, field: fld.Field) -> list[Optional[Action]]:
        if field.game_state not in [GameStates.KICKOFF, GameStates.PENALTY]:
            if field.active_team in [const.Color.ALL, field.ally_color]:
                self.we_active = True
            else:
                self.we_active = False

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

        # Вратарь
        if self.goalkeeper_id is not None:
            gk_target = self.get_goalkeeper_target(field)
            angle_to_ball = (curr_ball - gk_target).arg()
            actions[self.goalkeeper_id] = Actions.GoToPointIgnore(gk_target, angle_to_ball)

        # Нападающий
        if self.attacker_id is not None:
            actions[self.attacker_id] = self.get_attacker_action(field)

        self.prev_ball = curr_ball