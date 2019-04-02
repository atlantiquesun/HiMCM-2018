#by Jerry Yang
#!/usr/bin/ruby

class TempComputation
  # COP refers to COP of cooling.
  @@c_wall = 3000000
  @@c_room = 150000
  @@conduction_rate = 800
  @@interval = 15 * 60.0
  @@cop = 4.0
  @@alpha = 500.0
  @@beta = -2.0

  def self.get_dv_temp(temp, next_temp)
    (next_temp - temp) / @@interval
  end

  def self.power(temp, temp_wall, dv_temp)
    qr = @@conduction_rate * (temp_wall - temp) - @@c_room * dv_temp
    if qr >= 0 then
      return qr * @@interval / @@cop
    else
      return -qr * @@interval / (@@cop + 1)
    end
  end

  def self.utility(temp_state)
    @@alpha * temp_state.user.satisfaction(temp_state.temp, temp_state.time) * @@interval + @@beta * power(temp_state.temp, temp_state.temp_wall, temp_state.dv_temp)
  end

  def self.next_temp_wall(temp_room, temp_wall, temp_amb)
    temp_wall + (@@interval * @@conduction_rate * (temp_room + temp_amb - 2.0 * temp_wall)) / @@c_wall
  end
end

class UserState
  def initialize(left_time, pdf_return, temp_fav, stddev)
    @left_time = left_time
    @pdf_return = pdf_return
    @cdf_return = compute_cdf
    @temp_fav = temp_fav
    @stddev = stddev
  end

  def at_home?(time)
    if time < @left_time then
      return 1.0
    else
      return @cdf_return[time - @left_time]
    end
  end

  def satisfaction(temp, time)
    Math.exp(-(temp - @temp_fav[time])**2 / (2.0 * @stddev**2)) * at_home?(time)
  end

  private
  def compute_cdf
    cdf = [ @pdf_return[0] ]
    (1...@pdf_return.size).each do |i|
      cdf.push(cdf[-1] + @pdf_return[i])
    end
    return cdf
  end
end

class TempState
  attr_accessor :temp, :user, :time, :temp_wall, :dv_temp

  @@ambient = [11, 11, 11, 11, 11, 11, 11, 11,
               11, 11, 11, 11, 11, 11, 11, 11,
               11, 11, 11, 11, 12, 12, 12, 12,
               12, 12, 12, 12, 13, 13, 13, 13,
               14, 14, 15, 15, 15, 15, 16, 16,
               17, 17, 18, 18, 19, 19, 19, 19,
               20, 20, 21, 21, 22, 22, 22, 22,
               21, 21, 21, 21, 20, 20, 20, 20,
               19, 19, 19, 19, 18, 18, 17, 17,
               16, 16, 16, 16, 15, 15, 15, 15,
               14, 14, 14, 14, 13, 13, 13, 13,
               12, 12, 12, 12, 11, 11, 11, 11]

  def initialize(temp, time, user, temp_wall, dv_temp)
    @temp = temp
    @time = time
    @user = user
    @temp_wall = temp_wall
    @dv_temp = dv_temp
  end

  def legal_moves
    (-16..16).map { |i| @temp + i * 0.125 }.reject { |temp| temp < 16 or temp > 34 or (user.at_home?(@time).between?(0.001, 0.999) and @@ambient[@time] - temp > 10) }
  end
  
  def last_block?
    @time == 95
  end

  def next_state(temp)
    TempState.new(temp, time + 1, @user, TempComputation.next_temp_wall(temp, @temp_wall, @@ambient[@time]), TempComputation.get_dv_temp(@temp, temp))
  end
end

class MonteCarloTreeNode
  attr_accessor :q, :n, :state, :parent, :children

  def initialize(state, parent)
    @state = state
    @untried = state.legal_moves
    @children = []
    @parent = parent
    # results
    @q = 0
    # number of visits
    @n = 0
  end

  def rollout
    current_rollout = @state
    net_util = TempComputation.utility(current_rollout)
    until current_rollout.last_block? do
      current_rollout = current_rollout.next_state(rollout_policy(current_rollout.legal_moves))
      net_util += TempComputation.utility(current_rollout)
    end
    return net_util
  end
  
  def expand
    child = MonteCarloTreeNode.new(@state.next_state(@untried.pop), self)
    @children.push(child)
    return child
  end
  
  def fully_expanded?
    @untried.empty?
  end
  
  def back_propagate(net_util)
    @n += 1
    @q += net_util
    if @parent then
      @parent.back_propagate(net_util)
    end
  end
  
  def is_terminal_node?
    return @state.last_block?
  end
  
  def best_child(c_param = 50000000)
    @children.max_by { |c| (c.q.to_f / c.n.to_f) + c_param * Math.sqrt(Math.log(@n) / c.n) }
  end
  
  private
  def rollout_policy(moves)
    return moves.sample
  end
end

class MonteCarloTreeSearch
  attr_accessor :root
  @@simulations = 12000

  def initialize(root)
    @root = root
  end
  
  def best_action
    @@simulations.times do |i|
      leaf = tree_policy
      leaf.back_propagate(leaf.rollout)
    end
    @root.best_child(c_param = 0.0)
  end
  
  def output_children
    @root.output_children
  end
  
  private
  def tree_policy
    current_node = @root
    until current_node.is_terminal_node? do
      unless current_node.fully_expanded?
        return current_node.expand
      else
        current_node = current_node.best_child
      end
    end
    return current_node
  end
end

temp_fav = [19] * 96
user = UserState.new(100, [], temp_fav, 2.5)
search = MonteCarloTreeSearch.new(MonteCarloTreeNode.new(TempState.new(17, 0, user, 17, 0), nil))

ncost = 0
nsati = 0
nutil = 0
i = 1

until search.root.state.last_block?
  action = search.best_action
  ncost += TempComputation.power(search.root.state.temp, search.root.state.temp_wall, TempComputation.get_dv_temp(search.root.state.temp, action.state.temp))
  nsati += user.satisfaction(search.root.state.temp, search.root.state.time)
  nutil += TempComputation.utility(search.root.state)
  puts "Predicted temperature for time block #{i}: #{action.state.temp}"
  puts "Predicted wall temperature: #{action.state.temp_wall}"
  puts "Cumulative cost: #{ncost}"
  puts "Cumulative satisfaction: #{nsati}"
  puts "Cumulative utility: #{nutil}"
  puts "\n\n"
  action.parent.children.each do |c|
    if c != action then
      c = nil
    end
  end
  action.parent = nil
  search = MonteCarloTreeSearch.new(action)
  i += 1
end
