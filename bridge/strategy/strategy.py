"""High-level strategy code"""

# !v DEBUG ONLY
import math  # type: ignore  # noqa: F401
from time import time  # type: ignore  # noqa: F401
from typing import Optional

from bridge import const
from bridge.auxiliary import aux, fld, rbt  # type: ignore  # noqa: F401
from bridge.auxiliary.aux import get_line_intersection, get_tangent_points
from bridge.const import State as GameStates
from bridge.router.actions import (  # type: ignore  # noqa: F401
    Action,
    Actions,
    KickActions,
    StrategyActions,
)

class Strategy:
    """Main class of strategy"""

    def __init__(
        self,
    ) -> None:
        self.we_active = False
        self.prev_ball = aux.Point(0, 0)
    


    def process(self, field: fld.Field) -> list[Optional[Action]]:
        """Game State Management"""
        if field.game_state not in [GameStates.KICKOFF, GameStates.PENALTY]:
            if field.active_team in [const.Color.ALL, field.ally_color]:
                self.we_active = True
            else:
                self.we_active = False

        actions: list[Optional[Action]] = []
        for _ in range(const.TEAM_ROBOTS_MAX_COUNT):
            actions.append(None)

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
                # The router will automatically prevent robots from getting too close to the ball
                self.run(field, actions)
            case GameStates.BALL_PLACEMENT:
                pass
            case GameStates.DEBUG:
                pass

        return actions

    def run(self, field: fld.Field, actions: list[Optional[Action]]) -> None:
        """
        ONE ITERATION of strategy
        NOTE: robots will not start acting until this function returns an array of actions,
              if an action is overwritten during the process, only the last one will be executed)

        Examples of getting coordinates:
        - field.allies[8].get_pos(): aux.Point -   coordinates  of the 8th  robot from the allies
        - field.enemies[14].get_angle(): float - rotation angle of the 14th robot from the opponents

        - field.ally_goal.center: Point - center of the ally goal
        - field.enemy_goal.hull: list[Point] - polygon around the enemy goal area


        Examples of robot control:
        - actions[2] = Actions.GoToPoint(aux.Point(1000, 500), math.pi / 2)
                The robot number 2 will go to the point (1000, 500), looking in the direction π/2 (up, along the OY axis)

        - actions[3] = Actions.Kick(field.enemy_goal.center)
                The robot number 3 will hit the ball to 'field.enemy_goal.center' (to the center of the enemy goal)

        - actions[9] = Actions.BallGrab(0.0)
                The robot number 9 grabs the ball at an angle of 0.0 (it looks to the right, along the OX axis)
        """

        field.strategy_image.draw_line(aux.Point(300, 0), aux.Point(1000, 1000))
        vect1 = aux.Point(700, 1000)
        vect1s = aux.Point(300, 0)
        vect1e = aux.Point(1000, 1000)
        ang = vect1.arg()
        startVect = aux.Point(vect1s.x + 100 * math.cos(ang + math.pi / 2), vect1s.y + 100 * math.sin(ang + math.pi / 2))
        endVect = aux.Point(vect1e.x + 100 * math.cos(ang - math.pi / 2), vect1e.y + 100 * math.sin(ang - math.pi / 2))

        field.strategy_image.draw_circle(aux.Point(1000, 1000), (200, 100, 255), 100)
        field.strategy_image.draw_circle(aux.Point(300, 0), (200, 150, 255), 100)
        field.strategy_image.draw_line(startVect, endVect)

        # field.strategy_image.draw_poly(aux.Point())
        # print(field.allies[0].get_pos())
        # print(field.allies[0].get_angle())

        for i in range(len(field.allies)):  
            robot_pos = field.allies[i].get_pos()      
            robot_angle = field.allies[i].get_angle()   

        
            arrow_length = 150 

            
            end_look_x = robot_pos.x + arrow_length * math.cos(robot_angle)
            end_look_y = robot_pos.y + arrow_length * math.sin(robot_angle)
            end_look_point = aux.Point(end_look_x, end_look_y)

            
            # field.strategy_image.draw_line(robot_pos, end_look_point)
            # field.strategy_image.draw_poly([aux.Point(end_look_x + 40 * math.cos(robot_angle), 
            #                                         end_look_y + 40 * math.sin(robot_angle)),
            #                                 aux.Point(end_look_x - 40 * math.cos(robot_angle),
            #                                         end_look_y - 40 * math.sin(robot_angle)),
            #                                 aux.Point(robot_pos.x + 40 * math.cos(robot_angle),
            #                                         robot_pos.y + 40 * math.sin(robot_angle)),
            #                                 aux.Point(robot_pos.x - 40 * math.cos(robot_angle),
            #                                         robot_pos.y - 40 * math.sin(robot_angle)),
            #                                 aux.Point(robot_pos.x + 200 * math.cos(robot_angle),
            #                                         robot_pos.y + 200 * math.sin(robot_angle)), 
            #                         ])

            left_ang = robot_angle + math.pi / 2
            right_ang = robot_angle - math.pi / 2

            

            field.strategy_image.draw_poly([
                aux.Point(robot_pos.x + 20 * math.cos(left_ang), robot_pos.y + 20 * math.sin(left_ang)),   # Хвост-лево
                aux.Point(end_look_x + 20 * math.cos(left_ang), end_look_y + 20 * math.sin(left_ang)),   # Основание наконечника-лево
                aux.Point(end_look_x + 50 * math.cos(left_ang), end_look_y + 50 * math.sin(left_ang)),   # Крыло левое расширение
                aux.Point(robot_pos.x + 200 * math.cos(robot_angle), robot_pos.y + 200 * math.sin(robot_angle)), # Нос впереди (+200)
                aux.Point(end_look_x + 50 * math.cos(right_ang), end_look_y + 50 * math.sin(right_ang)), # Крыло правое расширение
                aux.Point(end_look_x + 20 * math.cos(right_ang), end_look_y + 20 * math.sin(right_ang)), # Основание наконечника-право
                aux.Point(robot_pos.x + 20 * math.cos(right_ang), robot_pos.y + 20 * math.sin(right_ang)), # Хвост-право
            ])

        
        for i in range(len(field.enemies)):  
            robot_pos = field.enemies[i].get_pos()      
            robot_angle = field.enemies[i].get_angle()   

        
            arrow_length = 150 

            
            end_look_x = robot_pos.x + arrow_length * math.cos(robot_angle)
            end_look_y = robot_pos.y + arrow_length * math.sin(robot_angle)
            end_look_point = aux.Point(end_look_x, end_look_y)

            
            # field.strategy_image.draw_line(robot_pos, end_look_point)
            # field.strategy_image.draw_poly([aux.Point(end_look_x + 40 * math.cos(robot_angle), 
            #                                         end_look_y + 40 * math.sin(robot_angle)),
            #                                 aux.Point(end_look_x - 40 * math.cos(robot_angle),
            #                                         end_look_y - 40 * math.sin(robot_angle)),
            #                                 aux.Point(robot_pos.x + 40 * math.cos(robot_angle),
            #                                         robot_pos.y + 40 * math.sin(robot_angle)),
            #                                 aux.Point(robot_pos.x - 40 * math.cos(robot_angle),
            #                                         robot_pos.y - 40 * math.sin(robot_angle)),
            #                                 aux.Point(robot_pos.x + 200 * math.cos(robot_angle),
            #                                         robot_pos.y + 200 * math.sin(robot_angle)), 
            #                         ])

            left_ang = robot_angle + math.pi / 2
            right_ang = robot_angle - math.pi / 2

            

            field.strategy_image.draw_poly([
                aux.Point(robot_pos.x + 20 * math.cos(left_ang), robot_pos.y + 20 * math.sin(left_ang)),   # Хвост-лево
                aux.Point(end_look_x + 20 * math.cos(left_ang), end_look_y + 20 * math.sin(left_ang)),   # Основание наконечника-лево
                aux.Point(end_look_x + 50 * math.cos(left_ang), end_look_y + 50 * math.sin(left_ang)),   # Крыло левое расширение
                aux.Point(robot_pos.x + 200 * math.cos(robot_angle), robot_pos.y + 200 * math.sin(robot_angle)), # Нос впереди (+200)
                aux.Point(end_look_x + 50 * math.cos(right_ang), end_look_y + 50 * math.sin(right_ang)), # Крыло правое расширение
                aux.Point(end_look_x + 20 * math.cos(right_ang), end_look_y + 20 * math.sin(right_ang)), # Основание наконечника-право
                aux.Point(robot_pos.x + 20 * math.cos(right_ang), robot_pos.y + 20 * math.sin(right_ang)), # Хвост-право
            ])

        posAR0, posAR1 = field.enemies[0].get_pos(), field.allies[0].get_pos()
        posER0, posER1 = field.enemies[1].get_pos(), field.allies[1].get_pos()
        field.strategy_image.draw_line(posAR0, posAR1)
        field.strategy_image.draw_line(posER0, posER1)
        p = get_line_intersection(posAR0, posAR1, posER0, posER1, "SS")
        if p is not None:
            field.strategy_image.draw_circle(p, (200, 150, 255), 100)

        # field.strategy_image.draw_rect

        robot_pos = field.allies[0].get_pos()
        obstacle_center = field.enemies[0].get_pos()
        mid_point = (robot_pos + obstacle_center) / 2
        obstacle_radius = 100

        tangents_to_enemy = get_tangent_points(obstacle_center, mid_point, obstacle_radius)
        tangents_to_ally  = get_tangent_points(robot_pos, mid_point, obstacle_radius)

        if len(tangents_to_enemy) == 2 and len(tangents_to_ally) == 2:
            en_left = tangents_to_enemy[0]
            en_right = tangents_to_enemy[1]
            
           
            al_left = tangents_to_ally[1]   
            al_right = tangents_to_ally[0]


            field.strategy_image.draw_line(al_left, en_right, (255, 100, 0)) 
            field.strategy_image.draw_line(al_right, en_left, (255, 100, 0))
            

            field.strategy_image.draw_line(al_left, en_left, (0, 255, 0))   
            field.strategy_image.draw_line(al_right, en_right, (0, 255, 0)) 


        # curr_ball = field.ball.get_pos()
        # # prev_ball = field.ball._pos_ 
        

        # target_x = 0.0 

        # dy = curr_ball.y - self.prev_ball.y
        # dx = curr_ball.x - self.prev_ball.x

        # if abs(dx) > 1e-5:

        #     predicted_y = curr_ball.y + (target_x - curr_ball.x) * dy / dx

        #     intercept_point = aux.Point(target_x, predicted_y)

        #     if (target_x > curr_ball.x and dx > 0) or (target_x < curr_ball.x and dx < 0):

        #         field.strategy_image.draw_line(curr_ball, intercept_point, (255, 0, 0))
        #         field.strategy_image.draw_circle(intercept_point, (0, 255, 255), 50)
        #     actions[0] = Actions.GoToPointIgnore(intercept_point, 0)
        # self.prev_ball = curr_ball

        
        # if not hasattr(self, 'prev_ball') or self.prev_ball is None:
        #     self.prev_ball = field.ball.get_pos()
        #
        # Пример вратаря
        # curr_ball = field.ball.get_pos()
        
        # our_goal = field.ally_goal 
        # target_x = our_goal.center.x + our_goal.eye_forw.x * 100 

        # dy = curr_ball.y - self.prev_ball.y
        # dx = curr_ball.x - self.prev_ball.x

        # intercept_point = aux.Point(target_x, our_goal.center.y)

        # if abs(dx) > 1e-5:
        #     predicted_y = curr_ball.y + (target_x - curr_ball.x) * dy / dx

        #     if (target_x > curr_ball.x and dx > 0) or (target_x < curr_ball.x and dx < 0):

        #         max_y = max(our_goal.up.y, our_goal.down.y)
        #         min_y = min(our_goal.up.y, our_goal.down.y)

        #         final_y = max(min_y, min(max_y, predicted_y))
                
        #         intercept_point = aux.Point(target_x, final_y)

        #         field.strategy_image.draw_line(curr_ball, intercept_point, (255, 0, 0))
        #         field.strategy_image.draw_circle(intercept_point, (0, 255, 255), 50)
        
        # else:
        #     max_y = max(our_goal.up.y, our_goal.down.y)
        #     min_y = min(our_goal.up.y, our_goal.down.y)
        #     ball_follow_y = max(min_y, min(max_y, curr_ball.y))
        #     intercept_point = aux.Point(target_x, ball_follow_y)

        # robot_index = 2

        # angle_to_ball = (curr_ball - intercept_point).arg()
        
        # actions[robot_index] = Actions.GoToPointIgnore(intercept_point, angle_to_ball)

        # self.prev_ball = curr_ball


        # # print(actions)

        # curr_ball = field.ball.get_pos()
        # our_goal = field.ally_goal 
        
        # target_x = our_goal.center.x + our_goal.eye_forw.x * 100 

        # vec_to_up = our_goal.up - curr_ball
        # vec_to_down = our_goal.down - curr_ball

        # bisector_dir = (vec_to_up.unity() + vec_to_down.unity()).unity()

        # if abs(bisector_dir.x) > 1e-5:
        #     t = (target_x - curr_ball.x) / bisector_dir.x
        #     bisector_y = curr_ball.y + bisector_dir.y * t
        # else:
        #     bisector_y = our_goal.center.y

        # intercept_point = aux.Point(target_x, bisector_y)

        # dy = curr_ball.y - self.prev_ball.y
        # dx = curr_ball.x - self.prev_ball.x

        # if abs(dx) > 1e-5 and ((target_x > curr_ball.x and dx > 0) or (target_x < curr_ball.x and dx < 0)):
        #     predicted_y = curr_ball.y + (target_x - curr_ball.x) * dy / dx
            
        #     max_y = max(our_goal.up.y, our_goal.down.y)
        #     min_y = min(our_goal.up.y, our_goal.down.y)
        #     final_y = max(min_y, min(max_y, predicted_y))
            
        #     intercept_point = aux.Point(target_x, final_y)

        #     field.strategy_image.draw_line(curr_ball, intercept_point, (255, 0, 0))
        # else:
        #     max_y = max(our_goal.up.y, our_goal.down.y)
        #     min_y = min(our_goal.up.y, our_goal.down.y)
        #     final_y = max(min_y, min(max_y, bisector_y))
            
        #     intercept_point = aux.Point(target_x, final_y)

        #     field.strategy_image.draw_line(curr_ball, our_goal.up, (0, 255, 0))
        #     field.strategy_image.draw_line(curr_ball, our_goal.down, (0, 255, 0))
        #     field.strategy_image.draw_line(curr_ball, intercept_point, (0, 255, 255))

        # field.strategy_image.draw_circle(intercept_point, (0, 255, 255), 50)

        # robot_index = 2
        # angle_to_ball = (curr_ball - intercept_point).arg()
        # actions[robot_index] = Actions.GoToPointIgnore(intercept_point, angle_to_ball)

        # self.prev_ball = curr_ball

        curr_ball = field.ball.get_pos()
        our_goal = field.ally_goal 
        arc_radius = 450.0

        vec_to_up = our_goal.up - curr_ball
        vec_to_down = our_goal.down - curr_ball

        bisector_dir = (vec_to_up.unity() + vec_to_down.unity()).unity()

        # arc_point = our_goal.center + bisector_dir * arc_radius
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

        robot_index = 2
        angle_to_ball = (curr_ball - intercept_point).arg()
        actions[robot_index] = Actions.GoToPointIgnore(intercept_point, angle_to_ball)

        self.prev_ball = curr_ball



