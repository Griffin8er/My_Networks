from manim import *
import numpy as np

#To run scripts, you must download manim and dependencies first
#Then, type in command into a terminal manim <script> <class> -pqm

class Move(Scene):
    def construct(self):
        box = Rectangle(stroke_color=GREEN_C, stroke_opacity=0.7,
                        fill_color=RED_B, fill_opacity=0.5, height=1, width=1)
        self.add(box)
        self.play(box.animate.shift(RIGHT*2), run_time=2)
        self.play(box.animate.shift(UP*3), run_time=2)
        self.play(box.animate.shift(DOWN*5+LEFT*5), run_time=2)
        self.play(box.animate.shift(UP*1.5+RIGHT*1), run_time=2)


class Objects(Scene):
    def construct(self):
        axes = Axes(x_range=[-3, 3, 1], y_range=[-3, 3, 1],
                    x_length = 6, y_length=6)
        axes.to_edge(LEFT, buff=0.5)

        circle = Circle(stroke_width=6, stroke_color=YELLOW,
                        fill_color=RED_C, fill_opacity=0.8)
        circle.set_width(2).to_edge(DR, buff=0)

        triangle=Triangle(stroke_color=ORANGE, stroke_width=10,
                          fill_color=GREY).set_height(2).shift(DOWN*3+RIGHT*3)
        self.play(Write(axes))
        self.play(DrawBorderThenFill(circle))
        self.play(circle.animate.set_width(1))
        self.play(Transform(circle, triangle), run_time=3)


class Updaters(Scene):
    def construct(self):
        rectangle = RoundedRectangle(stroke_width=8, stroke_color=WHITE, fill_color=BLUE_B,
                                     width=4.5, height=2).shift(UP*3+LEFT*4)
        
        math_text = MathTex("\\frac{3}{4} = 0.75").set_color_by_gradient(GREEN, PINK).set_height(1.5)
        math_text.move_to(rectangle.get_center())
        math_text.add_updater(lambda x : x.move_to(rectangle.get_center()))

        self.play(FadeIn(rectangle))
        self.play(Write(math_text))
        self.play(rectangle.animate.shift(RIGHT*1.5+DOWN*5), run_time=6)
        self.wait()
        math_text.clear_updaters()
        self.play(rectangle.animate.shift(LEFT*2 + UP*1), run_time=4)


class NewAnimation(Scene):
    def construct(self):
        r = ValueTracker(0.5)

        circle = always_redraw(lambda : Circle(radius=r.get_value(), stroke_color=YELLOW, stroke_width=5))

        line_radius = always_redraw(lambda : Line(start=circle.get_center(), end=circle.get_bottom(), stroke_color=RED_B, stroke_width=10))
        
        line_circ = always_redraw(lambda : Line(stroke_color=YELLOW, stroke_width=5).set_length(2*r.get_value()*PI).next_to(circle, DOWN, buff=0.2))

        triangle = always_redraw(lambda : Polygon(circle.get_top(), circle.get_left(), circle.get_right(), fill_color=GREEN_C))

        self.play(LaggedStart(Create(circle), DrawBorderThenFill(line_radius), DrawBorderThenFill(triangle), run_time=4, lag_ratio=0.75))
        self.play(ReplacementTransform(circle.copy(), line_circ), run_time=2)
        self.play(r.animate.set_value(2), run_time=5)

    
class Possibilities(Scene):
    def construct(self):

        circle = Circle(radius = .5, stroke_color=RED_C, stroke_width=3)

        self.play(Create(circle), run_time=2)
        self.play(circle.animate.shift(RIGHT*2+DOWN*4), run_time=4)
        self.play(circle.animate.shift(UP*4+LEFT*2), run_time=3)
        self.play(circle.animate(run_time=3).scale(4))
        self.play(Uncreate(circle), run_time=4)


class Graph(Scene):
    def construct(self):
        axes = Axes(x_range=[0, 3*PI, PI/2], y_range=[-2, 2, 1], x_length=3*PI, y_length=4,
                    axis_config={'include_tip': True, 'numbers_to_exclude':[0]})
        graph = axes.plot(lambda x : np.sin(x), x_range=[0, 2*PI], color=YELLOW)
        graph_stuff = VGroup(axes, graph)
        number_plane = NumberPlane(background_line_style={"stroke_opacity": 0.4})

        self.add(number_plane)
        self.play(DrawBorderThenFill(axes))
        self.play(Create(graph))
        self.play(graph_stuff.animate.shift(UP*1), run_time=2)
        self.play(graph_stuff.animate(run_time=3).scale(2))
        self.pause(1)
        self.play(graph_stuff.animate.shift(RIGHT*50), run_time=1)
        self.clear()

        circle = Circle(radius = .5, stroke_color=RED_C, stroke_width=3)
        self.add(number_plane)
        self.pause(2)
        self.play(Create(circle), run_time=2)
        self.play(circle.animate.shift(RIGHT*2+DOWN*4), run_time=4)
        self.play(circle.animate.shift(UP*4+LEFT*2), run_time=3)
        self.play(circle.animate(run_time=3).scale(4))
        self.play(Uncreate(circle), run_time=4)


class Brace_1(Scene):
    def construct(self):
        dot = Dot([0, 0, 1])
        dot2 = Dot([3, 3, 1])
        line = Line(dot.get_center(), dot2.get_center(), color=ORANGE).set_z_index(dot.z_index-1)
        bracket = Brace(line)
        bracket_name = bracket.get_text("Horizontal Distance")
        bracket2 = Brace(line, direction=line.copy().rotate(PI/2).get_unit_vector())
        bracket2_name = bracket2.get_tex("x-x_1")
        self.play(Create(dot))
        self.play(Create(dot2))
        self.play(Create(line), run_time=2)
        self.play(Create(bracket), Write(bracket_name), Create(bracket2), Write(bracket2_name), run_time=2)


config.background_color = WHITE
class Tangent_Line(Scene):
    def construct(self):
        dot = ValueTracker(-1)
        axis = Axes([-6, 6, 1], [-4, 4, 1], x_length=8, y_length=6).set_color(BLACK)
        x_graph = axis.plot(lambda x : -(x)**3 + 3*(x)**2, x_range=[-1, (10.07)/3], color=ORANGE)
        line = always_redraw(lambda : axis.plot(lambda x : (((-3*(dot.get_value())**2 + 6*(dot.get_value()))*(x-dot.get_value()))+
                                                    (-(dot.get_value())**3 + 3*(dot.get_value())**2)), x_range=(dot.get_value()-1, dot.get_value()+1), stroke_color=BLACK))
        
        self.play(DrawBorderThenFill(axis), run_time=2)
        self.play(Create(x_graph), run_time=2)
        self.play(Create(line))
        self.play(dot.animate.set_value(2), run_time=3)
        box = Rectangle(height=1/3, width=2, stroke_color=BLACK, 
                        stroke_width=4).set_fill(BLUE_D, 1).next_to(line, UP, buff=0.0)
        self.play(ReplacementTransform(line, box), run_time=1)
        self.play(box.animate.rotate(90*DEGREES).next_to([0, 0, 0], UR, buff=0.0))
        self.play(box.animate.scale([1, 0, 1]).shift(DOWN*1))
        for i in range(1, 6, 1):
            new_box=Rectangle(height=(3/4)*(-(i/2)**3 + 3*(i/2)**2), width=1/3, stroke_color=BLACK).next_to([1/3*i, 0, 0], UR, buff=0)
            self.play(Create(new_box), run_time=.5)

        self.wait(1)


class Third_Dim(ThreeDScene):
    def construct(self):
        axis = ThreeDAxes([-6, 6, 1], [-6, 6, 1], [-6, 6, 1], 8, 6, 6).set_color(BLACK)
        graph = ParametricFunction(lambda t : np.array([np.cos(t), np.sin(3*t), np.cos(5*t)]), t_range=[-TAU, TAU], color=ORANGE)

        self.play(Create(axis))
        self.move_camera(phi=60*DEGREES, theta=45*DEGREES)
        self.play(Create(graph))
        self.begin_ambient_camera_rotation(rate=.2, about="phi")
        self.begin_ambient_camera_rotation(rate=.3)
        self.wait(4)