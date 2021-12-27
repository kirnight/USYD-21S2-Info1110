import config
from space_object import SpaceObject
import random
import math
import numpy as np

class Engine:
    def __init__(self, game_state_filename, player_class, gui_class):
        self.import_state(game_state_filename)
        self.player = player_class()
        self.GUI = gui_class(self.width, self.height)
        self.asteroid_large_num = 0
        self.asteroid_small_num = 0
        self.switch = 0
        self.bullet_index_counter = 0
        self.bullet_travelled = []
        self.challenge_switch = 0
        self.timer = 0
        self.time_count = 1
        self.collided_asteroids = 0
        self.bullet_hitted_asteroids = 0
        self.initial_asteroid = 0

 
    def raise_error(self, line , line_num , expected_key):
        # Creating a bunch of code which checks the value that is just a number(an int)
        if line == "":
            raise ValueError('Error: game state incomplete')
        try:
            if line.split(" ")[1] == "":
                raise ValueError("Error: expecting a key and value in line {}".format(line_num))
        except IndexError:
            pass        
        try:
            if line.split(" ")[2] != "":
                raise ValueError("Error: expecting a key and value in line {}".format(line_num))
        except IndexError:
            pass
        if line.split(" ")[0] != expected_key:
            raise ValueError("Error: unexpected key: {} in line {}".format(line.split(" ")[0] , line_num))
        try:
            int(line.split(" ")[1])
        except ValueError:
            raise ValueError("Error: invalid data type in line {}".format(line_num))
        

    def spaceobject_error(self , space_object_data , line_num):
        # Creating a bunch of code that checks the value for space objects. Actually, I think it is useless, but checking in case some edge case happen.
        try:
            if space_object_data.split(",")[4] != "":
                raise ValueError("Error: invalid data type in line {}".format(line_num))
        except IndexError:
            pass
        try:
            int(space_object_data.split(",")[2])
            int(space_object_data.split(",")[3])
        except ValueError:
            raise ValueError("Error: invalid data type in line {}".format(line_num))
        try:
            float(space_object_data.split(",")[0])
            float(space_object_data.split(",")[1])
        except ValueError:
            raise ValueError("Error: invalid data type in line {}".format(line_num))
        if space_object_data.split(",")[0].isdigit() or space_object_data.split(",")[1].isdigit():
            raise ValueError("Error: invalid data type in line {}".format(line_num))


    def import_state(self, game_state_filename):
        # Modularize the game state into four parts the basic part for width, height, score, spaceship, fuel, and the other three parts which are the asteroids, bullets and upcoming asteroids.
        # This mustn't be the best way, but there's no time leaving for me to figure out a faster way.
        
        self.asteroid_small_num = 0
        self.asteroid_large_num = 0
        try:
            line_num = 0
            number = 0
            game_state_file = open(game_state_filename , "r")
            self.basic = {}
            self.asteroid_ls = []
            self.bullet_ls = []
            self.asteroids = []
            self.upcoming_asteroids = []
        except FileNotFoundError:
            raise FileNotFoundError("Error: unable to open {}".format(game_state_filename))

        # For width, height, score, spaceship, fuel
        while True:
            line = game_state_file.readline().strip()
            line_num += 1
            if line == "":
                raise ValueError('Error: game state incomplete')
            # Width
            if line_num == 1:
                self.raise_error(line,line_num,"width")
                self.width = int(line.split(" ")[1])
                self.basic["width"] = self.width
            # Height
            if line_num == 2:
                self.raise_error(line,line_num,"height")
                self.height = int(line.split(" ")[1])                 
                self.basic["height"] = self.height
            # Score
            if line_num == 3:
                self.raise_error(line,line_num,"score")
                self.score = int(line.split(" ")[1])
            # Spaceship
            if line_num == 4:
                try:
                    if line.split(" ")[1] == "":
                        raise ValueError("Error: expecting a key and value in line {}".format(line_num))
                except IndexError:
                    pass                
                try:
                    if line.split(" ")[2] != "":
                        raise ValueError("Error: expecting a key and value in line {}".format(line_num))
                except IndexError:
                    pass
                if line.split(" ")[0] != "spaceship":
                    raise ValueError("Error: unexpected key: {} in line {}".format(line.split(" ")[0] , line_num))   
                space_object_data = line.split(" ")[1]
                try:
                    self.spaceship = SpaceObject(space_object_data.split(",")[0] , space_object_data.split(",")[1] , self.width , self.height , space_object_data.split(",")[2] , line.split(" ")[0] , space_object_data.split(",")[3])
                except IndexError:
                    raise ValueError("Error: invalid data type in line {}".format(line_num))
                except ValueError:
                    raise ValueError("Error: invalid data type in line {}".format(line_num))  
                self.spaceobject_error(space_object_data,line_num)
            # Fuel
            if line_num == 5:
                self.raise_error(line,line_num,"fuel")
                self.fuel = int(line.split(" ")[1])
                break
        
        switch1 = 0
        switch2 = 0
        switch3 = 0
        asteriod_id = 0
        while True:
            line = game_state_file.readline().strip()
            line_num += 1

            if switch1 == 1 and switch2 == 1 and switch3 == 1 and line == "":
                break

            elif line == "":
                raise ValueError('Error: game state incomplete')

            # For asteroids
            elif line.split(" ")[0] == "asteroids_count":
                self.raise_error(line,line_num,"asteroids_count")
                number = int(line.split(" ")[1])
                count = 0
                while count < number:
                    while True:
                        asteriod_x = random.randint(0,self.width)
                        asteriod_y = random.randint(0,self.height)
                        if asteriod_x != self.spaceship.x and asteriod_y != self.spaceship.y:
                            break
                    asteroid_type = random.choice(["asteroid_large","asteroid_small"])
                    if asteroid_type == "asteroid_large":
                        self.asteroid_large_num += 1
                    if asteroid_type == "asteroid_small":
                        self.asteroid_small_num += 1
                    self.asteroid_ls.append(SpaceObject(asteriod_x , asteriod_y , self.width , self.height , random.randint(0,360) , asteroid_type , asteriod_id))
                    asteriod_id += 1
                    count += 1
                switch1 = 1
            
            # For bullets
            elif line.split(" ")[0] == "bullets_count":
                self.raise_error(line,line_num,"bullets_count")
                number = int(line.split(" ")[1])
                count = 0
                while count < number:
                    line = game_state_file.readline().strip()
                    line_num += 1
                    if line == "":
                        raise ValueError('Error: game state incomplete')
                    if line.split(" ")[0] != "bullet":
                        raise ValueError("Error: unexpected key: {} in line {}".format(line.split(" ")[0] , line_num))  
                    try:
                        space_object_data = line.split(" ")[1]
                    except IndexError:
                        raise ValueError("Error: expecting a key and value in line {}".format(line_num))
                    try:
                        if line.split(" ")[2] != "":
                            raise ValueError("Error: expecting a key and value in line {}".format(line_num))
                    except IndexError:
                        pass
                    try:
                        self.bullet_ls.append(SpaceObject(space_object_data.split(",")[0] , space_object_data.split(",")[1] , self.width , self.height , space_object_data.split(",")[2] , line.split(" ")[0] , space_object_data.split(",")[3]))
                    except IndexError:
                        raise ValueError("Error: invalid data type in line {}".format(line_num))
                    except ValueError:
                        raise ValueError("Error: invalid data type in line {}".format(line_num))  
                    self.spaceobject_error(space_object_data,line_num)
                    count += 1
                switch2 = 1

            # For upcoming asteroids
            elif line.split(" ")[0] == "upcoming_asteroids_count":
                self.raise_error(line,line_num,"upcoming_asteroids_count")
                number = int(line.split(" ")[1])
                count = 0
                while count < number:
                    asteriod_x = random.randint(0,self.width)
                    asteriod_y = random.randint(0,self.height)
                    asteroid_type = random.choice(["asteroid_large","asteroid_small"])
                    self.upcoming_asteroids.append(SpaceObject(asteriod_x , asteriod_y , self.width , self.height , random.randint(0,360) , asteroid_type , asteriod_id))
                    asteriod_id += 1
                    count += 1
                switch3 = 1

            else:
                raise ValueError("Error: unexpected key: {} in line {}".format(line.split(" ")[0] , line_num)) 

        game_state_file.close()


    def export_state(self, game_state_filename):
        game_state_file = open(game_state_filename , "w")
        line_num = 0

        # For width, height, score, spaceship, fuel
        while True:
            line_num += 1
            # Width
            if line_num == 1:
                line = "width {}\n".format(self.basic["width"])
                game_state_file.write(line)
            # Height
            if line_num == 2:
                line = "height {}\n".format(self.basic["height"])
                game_state_file.write(line)
            # Score
            if line_num == 3:
                line = "score {}\n".format(self.score)
                game_state_file.write(line)
            # Spaceship
            if line_num == 4:
                line = "{}\n".format(repr(self.spaceship))
                game_state_file.write(line)
            # Fuel
            if line_num == 5:
                line = "fuel {}\n".format(self.fuel)
                game_state_file.write(line)
                break
        
        # For asteroids       
        line = "asteroids_count {}\n".format(len(self.asteroid_ls))
        game_state_file.write(line)
        line_num += 1
        count = 0
        while count < len(self.asteroid_ls):
            line_num += 1
            line = "{}\n".format(self.asteroid_ls[count].__repr__())
            game_state_file.write(line)
            count += 1
        
        # For bullets
        line = "bullets_count {}\n".format(len(self.bullet_ls))
        game_state_file.write(line)        
        line_num += 1
        count = 0
        while count < len(self.bullet_ls):
            line_num += 1
            line = "{}\n".format(self.bullet_ls[count].__repr__())
            game_state_file.write(line)
            count += 1

        # For upcoming asteroids          
        line = "upcoming_asteroids_count {}\n".format(len(self.upcoming_asteroids))
        game_state_file.write(line)
        line_num += 1
        count = 0
        while count < len(self.upcoming_asteroids):
            line_num += 1
            line = "upcoming_{}\n".format(self.upcoming_asteroids[count].__repr__())
            game_state_file.write(line)
            count += 1

        game_state_file.close()


    def state_tran(self):
        array_len = 5 + 5 * len(self.asteroid_ls)
        state = np.zeros(array_len)
        state[0] = self.spaceship.x
        state[1] = self.spaceship.y
        state[2] = self.spaceship.angle
        state[3] = 0
        state[4] = self.spaceship.id
        index = 5
        for x in self.asteroid_ls:
            state[index] = x.x
            index += 1
            state[index] = x.y
            index += 1
            state[index] = x.angle
            index += 1
            if "small" in x.obj_type:
                state[index] = 0
            if "large" in x.obj_type:
                state[index] = 1
            index += 1
            state[index] = x.id
            index += 1
        return state


    def data_tran(self):
        return[self.bullet_hitted_asteroids , self.collided_asteroids , self.fuel , self.score , self.time_count]


    def train(self, action):
        # 1. Receive player input
        player_input = action

        self.initial_asteroid = len(self.asteroid_ls)

        # reset bullet count
        self.bullet_hitted_asteroids = 0
        self.collided_asteroids = 0

        # 2. Process game logic
        ## Manoeuvre the spaceship as per the Player's input
        if not (player_input[1] and player_input[2]):
            if player_input[1]:
                self.spaceship.turn_left()
            elif player_input[2]:
                self.spaceship.turn_right()
        if player_input[0]:
            self.spaceship.move_forward()

        ## Update positions of asteroids by calling move_forward() for each asteroid
        for x in self.asteroid_ls:
            if self.score <= 0:
                min_speed = 0
                max_speed = config.speed[x.obj_type]
                speed = random.randint(min_speed,max_speed)

                dx = math.cos(math.radians(x.angle))
                dy = math.sin(math.radians(x.angle))
                x.x += speed * dx
                x.y -= speed * dy
                SpaceObject.check_xy(x)

            if self.score > 0:
                min_speed = 2 * round(pow(math.log10(abs(self.score + 100)) , 2)) - 6 + 0
                max_speed = 2 * round(pow(math.log10(abs(self.score + 100)) , 2)) - 6 + config.speed[x.obj_type]
                speed = random.randint(min_speed,max_speed)

                dx = math.cos(math.radians(x.angle))
                dy = math.sin(math.radians(x.angle))
                x.x += speed * dx
                x.y -= speed * dy
                SpaceObject.check_xy(x)

        ## Update positions of bullets
        if player_input[3]:
            if self.fuel >= config.shoot_fuel_threshold:
                self.bullet_ls.append(SpaceObject(self.spaceship.x , self.spaceship.y , self.width , self.height , self.spaceship.angle , "bullet" , self.bullet_index_counter))
                self.bullet_travelled.append(0)
                self.bullet_index_counter += 1
                self.fuel -= config.bullet_fuel_consumption

        i = 0
        while i < len(self.bullet_ls):
            self.bullet_ls[i].move_forward()
            self.bullet_travelled[i] += 1
            if self.bullet_travelled[i] > config.bullet_move_count:
                del self.bullet_ls[i]
                del self.bullet_travelled[i]
                i -= 1
            i += 1
        
        ## Deduct fuel for spaceship and bullets (if launched)
        self.fuel -= config.spaceship_fuel_consumption

        ## Detect collisions
        ### bullet vs asteroid
        collision_bullet_counter = 0
        for x in self.bullet_ls:
            collision_asteroid_counter = 0
            for y in self.asteroid_ls:
                result = x.collide_with(y)
                if result:
                    if y.__repr__().split(" ")[0] == "asteroid_small":
                        self.score += config.shoot_small_ast_score
                    if y.__repr__().split(" ")[0] == "asteroid_large":
                        self.score += config.shoot_large_ast_score
                    del self.bullet_ls[collision_bullet_counter]
                    del self.bullet_travelled[collision_bullet_counter]
                    del self.asteroid_ls[collision_asteroid_counter]
                    self.bullet_hitted_asteroids += 1
                collision_asteroid_counter += 1
                if result:
                    break
            collision_bullet_counter += 1

        ### spaceship vs asteroid
        collision_asteroid_counter = 0
        for x in self.asteroid_ls:
            result = self.spaceship.collide_with(x)
            if result:
                self.score += config.collide_score
                del self.asteroid_ls[collision_asteroid_counter]
                self.collided_asteroids += 1
            collision_asteroid_counter += 1

        ### add asteroid
        while True:
            if len(self.asteroid_ls) < self.initial_asteroid:
                if self.upcoming_asteroids != []:
                    self.asteroid_ls.append(self.upcoming_asteroids[0])
                    del self.upcoming_asteroids[0]
                if self.upcoming_asteroids == []:
                    self.switch = 1
                    break
            if len(self.asteroid_ls) == self.initial_asteroid:
                break

        self.timer += 1
        self.time_count += 1

        if self.switch == 1:
            return True
        if self.fuel == 0:
            return True
        else:
            return False


    def test(self, action):
        # 1. Receive player input
        player_input = action

        self.initial_asteroid = len(self.asteroid_ls)

        # set bullet count
        self.bullet_hitted_asteroids = 0
        self.collided_asteroids = 0


        # 2. Process game logic
            ## Manoeuvre the spaceship as per the Player's input
        if not (player_input[1] and player_input[2]):
            if player_input[1]:
                self.spaceship.turn_left()
            elif player_input[2]:
                self.spaceship.turn_right()
        if player_input[0]:
            self.spaceship.move_forward()

        ## Update positions of asteroids by calling move_forward() for each asteroid
        for x in self.asteroid_ls:
            if self.score <= 0:
                min_speed = 0
                max_speed = config.speed[x.obj_type]
                speed = random.randint(min_speed,max_speed)

                dx = math.cos(math.radians(x.angle))
                dy = math.sin(math.radians(x.angle))
                x.x += speed * dx
                x.y -= speed * dy
                SpaceObject.check_xy(x)

            if self.score > 0:
                min_speed = 2 * round(pow(math.log10(abs(self.score + 100)) , 2)) - 6 + 0
                max_speed = 2 * round(pow(math.log10(abs(self.score + 100)) , 2)) - 6 + config.speed[x.obj_type]
                speed = random.randint(min_speed,max_speed)

                dx = math.cos(math.radians(x.angle))
                dy = math.sin(math.radians(x.angle))
                x.x += speed * dx
                x.y -= speed * dy
                SpaceObject.check_xy(x)

        ## Update positions of bullets
        if player_input[3]:
            if self.fuel >= config.shoot_fuel_threshold:
                self.bullet_ls.append(SpaceObject(self.spaceship.x , self.spaceship.y , self.width , self.height , self.spaceship.angle , "bullet" , self.bullet_index_counter))
                self.bullet_travelled.append(0)
                self.bullet_index_counter += 1
                self.fuel -= config.bullet_fuel_consumption
            elif self.fuel < config.shoot_fuel_threshold:
                print("Cannot shoot due to low fuel")
                pass
        

        i = 0
        while i < len(self.bullet_ls):
            self.bullet_ls[i].move_forward()
            self.bullet_travelled[i] += 1
            if self.bullet_travelled[i] > config.bullet_move_count:
                del self.bullet_ls[i]
                del self.bullet_travelled[i]
                i -= 1
            i += 1
        
        ## Deduct fuel for spaceship and bullets (if launched)
        self.fuel -= config.spaceship_fuel_consumption

        ## Detect collisions
        ### bullet vs asteroid
        collision_bullet_counter = 0
        for x in self.bullet_ls:
            collision_asteroid_counter = 0
            for y in self.asteroid_ls:
                result = x.collide_with(y)
                if result:
                    if y.__repr__().split(" ")[0] == "asteroid_small":
                        self.score += config.shoot_small_ast_score
                    if y.__repr__().split(" ")[0] == "asteroid_large":
                        self.score += config.shoot_large_ast_score
                    print("Score: {} \t [Bullet {} has shot asteroid {}]".format(self.score,x.__repr__().split(" ")[1].split(",")[3],y.__repr__().split(" ")[1].split(",")[3]))
                    del self.bullet_ls[collision_bullet_counter]
                    del self.bullet_travelled[collision_bullet_counter]
                    del self.asteroid_ls[collision_asteroid_counter]
                collision_asteroid_counter += 1
                if result:
                    break
            collision_bullet_counter += 1

        ### spaceship vs asteroid
        collision_asteroid_counter = 0
        for x in self.asteroid_ls:
            result = self.spaceship.collide_with(x)
            if result:
                self.score += config.collide_score
                print("Score: {} \t [Spaceship collided with asteroid {}]".format(self.score,x.__repr__().split(" ")[1].split(",")[3]))
                del self.asteroid_ls[collision_asteroid_counter]
                self.collided_asteroids += 1
            collision_asteroid_counter += 1


        ### add asteroid
        while True:
            if len(self.asteroid_ls) < self.initial_asteroid:
                if self.upcoming_asteroids != []:
                    self.asteroid_ls.append(self.upcoming_asteroids[0])
                    print("Added asteroid {}".format(self.upcoming_asteroids[0].__repr__().split(" ")[1].split(",")[3]))
                    del self.upcoming_asteroids[0]
                if self.upcoming_asteroids == []:
                    print ("Error: no more asteroids available")
                    self.switch = 1
                    break
            if len(self.asteroid_ls) == self.initial_asteroid:
                break

        self.timer += 1
        self.time_count += 1

        # 3. Draw the game state on screen using the GUI class
        self.GUI.update_frame(self.spaceship, self.asteroid_ls, self.bullet_ls, self.score, self.fuel)


        if self.switch == 1:
            return True
        if self.fuel == 0:
            return True
        else:
            return False

