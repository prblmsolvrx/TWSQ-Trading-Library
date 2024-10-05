<!DOCTYPE html>
<html>

<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TWSQDocs</title>
  <link rel="stylesheet" href="https://stackedit.io/style.css" />
</head>

<body class="stackedit">
  <div class="stackedit__left">
    <div class="stackedit__toc">
      
<ul>
<li><a href="#twsq-trading-library">TWSQ Trading Library</a></li>
<li><a href="#introduction">Introduction</a>
<ul>
<li><a href="#installation">Installation</a></li>
<li><a href="#set-the-twsqroot-directory">Set the TWSQROOT Directory</a></li>
<li><a href="#create-a-strategy">Create a Strategy</a></li>
<li><a href="#backtestinga-idbacktest-introa">Backtesting</a></li>
<li><a href="#live-trading">Live Trading</a></li>
</ul>
</li>
<li><a href="#strategy-construction">Strategy Construction</a>
<ul>
<li><a href="#rebalance">Rebalance</a></li>
<li><a href="#order-management">Order Management</a></li>
<li><a href="#pricevolume-data">Price/Volume Data</a></li>
<li><a href="#strategy-parameters-and-preparation">Strategy Parameters and Preparation</a></li>
<li><a href="#get-positions">Get Positions</a></li>
<li><a href="#custom-data">Custom Data</a></li>
<li><a href="#exiting">Exiting</a></li>
</ul>
</li>
<li><a href="#backtesting">Backtesting</a>
<ul>
<li><a href="#launching">Launching</a></li>
<li><a href="#core-backtesting-logic">Core Backtesting Logic</a></li>
<li><a href="#pnl-calculation">PnL Calculation</a></li>
<li><a href="#shorting">Shorting</a></li>
<li><a href="#output">Output</a></li>
</ul>
</li>
<li><a href="#live-trading-1">Live Trading</a>
<ul>
<li><a href="#configure-settings-file">Configure Settings File</a></li>
<li><a href="#launching-1">Launching</a></li>
<li><a href="#exchange-api-set-up">Exchange API Set Up</a></li>
<li><a href="#exiting-1">Exiting</a></li>
<li><a href="#output-1">Output</a></li>
<li><a href="#position-tracking">Position Tracking</a></li>
<li><a href="#telegram-logging">Telegram Logging</a></li>
</ul>
</li>
<li><a href="#easy-price-data">Easy Price Data</a></li>
</ul>

    </div>
  </div>
  <div class="stackedit__right">
    <div class="stackedit__html">
      <h1 id="twsq-trading-library">TWSQ Trading Library</h1>
<p>This library allows for backtesting and live trading fully automated Quant strategies in cryptocurrencies.</p>
<p>It specializes in deploying strategies that are based on <em>price-volume</em> data. Quant strategies that exploit price-volume patterns, commonly known as <em>statistical arbitrage</em>, are some of the strongest that exist. And crypto markets, being less efficient, should be fertile grounds for such strategies.</p>
<p>Live-trading is currently done on Kraken.</p>
<p>We are working to add additional exchanges, datasets, and asset classes. If you’d like to incorporate additional (non price-volume) data immediately, you will have to do some coding but it should be doable. See <a href="#custom-data">custom data</a>.</p>
<p>The rest of this guide should ideally be read in order by starting with the introduction and then proceeding to the other sections.</p>
<h1 id="introduction">Introduction</h1>
<h2 id="installation">Installation</h2>
<p>First download the <code>twsq</code> codebase. To install, from your terminal (or command prompt in windows), <code>cd</code> into the downloaded folder and run the <code>setup.py</code> file there with this command: <code>python setup.py install</code>.</p>
<p>For more advanced users who would like to build on top of this library, install using <code>python setup.py develop</code> instead. This allows any changes you make to the code to be immediately reflected (IE. no need to re-install the package again).</p>
<h2 id="set-the-twsqroot-directory">Set the TWSQROOT Directory</h2>
<p>The library needs a location to save down data and results. By default, it will create and use a folder called <code>MyTWSQ</code> in your home directory.</p>
<p>You can also set a custom location by setting an environment variable named <code>TWSQROOT</code>. For example, if <code>TWSQROOT</code> is set to<code>/User1/codebase/</code> the library would create and use a folder <code>/User1/codebase/MyTWSQ</code>. <a href="https://chlee.co/how-to-setup-environment-variables-for-windows-mac-and-linux/">Here</a> is a guide on how to create environment variables.</p>
<p>You can verify your TWSQROOT location like so:</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">from</span> twsq<span class="token punctuation">.</span>utils <span class="token keyword">import</span> get_twsqroot
get_twsqroot<span class="token punctuation">(</span><span class="token punctuation">)</span>
</code></pre>
<p>More on this later, but some of the data saved includes:</p>
<ol>
<li>Price-volume cache files (for faster retrieval)</li>
<li>Backtest / live trading results (positions, PnL, etc.)</li>
<li>Log files</li>
</ol>
<h2 id="create-a-strategy">Create a Strategy</h2>
<p>To create a strategy, first import the <code>Alpha</code> class from <code>twsq.alpha</code>.</p>
<p>You will then create your strategy as a subclass of the <code>Alpha</code> class. Your  subclass should define and fill out key functions that tell it what to do. In doing so, it can leverage pre-built functions of the parent <code>Alpha</code> class.</p>
<p>Here’s a simple strategy that buys 1 ETH/USD on each rebalance (IE. “dollar cost average” into ETH):</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">from</span> twsq<span class="token punctuation">.</span>alpha <span class="token keyword">import</span> Alpha

<span class="token keyword">class</span> <span class="token class-name">ETHDCA</span><span class="token punctuation">(</span>Alpha<span class="token punctuation">)</span><span class="token punctuation">:</span>

    <span class="token keyword">def</span> <span class="token function">rebalance</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span><span class="token punctuation">:</span>
        self<span class="token punctuation">.</span>create_order<span class="token punctuation">(</span>
            <span class="token string">'ETH/USD'</span><span class="token punctuation">,</span>
            <span class="token number">1</span><span class="token punctuation">,</span>
            <span class="token string">'buy'</span><span class="token punctuation">,</span>
            route<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">)</span>
            
    <span class="token keyword">return</span>
</code></pre>
<p>Above we’ve filled out the <code>rebalance</code> key function, which is the most important function to define for each strategy. This function gets executed at a certain frequency (IE. every hour, day, etc.) you can set when the backtester or live trader is run. As the name suggests, it’s used to rebalance your portfolio and generate new trades.</p>
<p>Within <code>rebalance</code>, we’ve leveraged the pre-built function <code>create_order</code> of the <code>Alpha</code> parent class to send an order to the broker.</p>
<p>Throughout the guide we discuss other key functions you can fill out as well as other pre-built functions of the <code>Alpha</code> class you can use.</p>
<p><strong>A final note: you only have to code this strategy subclass once and you can use it for both backtesting and live trading</strong></p>
<h2 id="backtestinga-idbacktest-introa">Backtesting<a id="backtest-intro"></a></h2>
<p>Here’s how you can backtest the <code>ETHDCA</code> example above:</p>
<pre class=" language-python"><code class="prism  language-python">result <span class="token operator">=</span> ETHDCA<span class="token punctuation">.</span>run_backtest<span class="token punctuation">(</span>start_ts<span class="token operator">=</span><span class="token string">'20210101'</span><span class="token punctuation">)</span>
</code></pre>
<p>Note that you don’t have to instantiate ETHDCA.</p>
<p>The backtest results (Trades, PnL, Positions) will automatically be stored in MyTWSQ/alphas/ETHDCA/backtest as two csv files:</p>
<ol>
<li>pos_pnl.csv</li>
<li>orders.csv</li>
</ol>
<p>You can also access these through the function output <code>result</code>.</p>
<h2 id="live-trading">Live Trading</h2>
<p>To live trade the <code>ETHDCA</code> strategy, you can simply run <code>ETHDCA.run_live()</code>. Note that you will have to first set up the broker and configure your settings file, discussed <a href="#configure-settings-file">below</a>.</p>
<h1 id="strategy-construction">Strategy Construction</h1>
<p>Please first see the <a href="#create-a-strategy">introduction</a> for how to create a basic strategy. As mentioned there, <em>the name of the game is essentially filling out key functions that are called in structured ways in the background while leveraging the pre-built functions of</em> <code>Alpha</code>.</p>
<p>Through this section, we introduce several more such functions including:</p>
<p>Key functions you can fill out:</p>
<ol>
<li><a href="#rebalance">rebalance</a> - rebalance your portfolio</li>
<li><a href="#strategy-parameters-and-preparation">prepare</a> - set strategy parameters &amp; prepare for trading</li>
<li><a href="#react-to-finished-orders">on_finished_order</a> - react to finished orders</li>
<li><a href="#on_exit">on_exit</a> - custom function run when trader exits</li>
</ol>
<p>Pre-built functions you can use when filling out key functions above:</p>
<ol>
<li><a href="#create_order">create_order</a> - create an order</li>
<li><a href="#trade_to_target">trade_to_target</a> - trade to specified target positions</li>
<li><a href="#cancel_all_orders">cancel_all_orders</a> - cancel all orders</li>
<li><a href="#cancel_order">cancel_order</a> - cancel a specific order</li>
<li><a href="#get-open-orders">get_open_orders</a> - get a list of open orders</li>
<li><a href="#get_current_price">get_current_price</a> - get the current price of a symbol</li>
<li><a href="#get_lastn_bars">get_lastn_bars</a> - get the last n OHLCV bars of a symbol</li>
<li><a href="#get-positions">get_pos</a> - get your current positions</li>
</ol>
<h2 id="rebalance">Rebalance</h2>
<p>We already discussed the <code>rebalance</code> function in the  <a href="#create-a-strategy">introduction</a>. To emphasize, this is the most important function to fill out. It rebalances your portfolio at a certain frequency (default = 1h) you set when you call <a href="#run-backtest">run_backtest</a> or <a href="#run-live">run_live</a>. This is where you should generate a proprietary signal and send trades according to it.</p>
<p>Below is pseudo-code for how rebalance gets called when you launch the backtester or live trader:</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">while</span> is_trading<span class="token punctuation">:</span>
    rebalance<span class="token punctuation">(</span><span class="token punctuation">)</span>
    sleep<span class="token punctuation">(</span>freq<span class="token punctuation">)</span>
</code></pre>
<h2 id="order-management">Order Management</h2>
<p>In this subsection, we review how you can <a href="#creating-orders">create orders</a>, <a href="#canceling-orders">cancel orders</a>, <a href="#get-open-orders">get your open orders</a>, and <a href="#react-to-finished-orders">react to finished orders</a></p>
<h3 id="creating-orders">Creating Orders</h3>
<p>The <code>Alpha</code> class has 2 pre-built functions for creating orders:</p>
<ol>
<li><code>self.create_order</code></li>
<li><code>self.trade_to_target</code></li>
</ol>
<p>You can call <code>self.create_order</code> within any of the functions you fill out. Most commonly, you will call it within the <code>rebalance</code> function as we did in the introductory example with <code>ETHDCA</code>, copied below.</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">from</span> twsq<span class="token punctuation">.</span>alpha <span class="token keyword">import</span> Alpha

<span class="token keyword">class</span> <span class="token class-name">ETHDCA</span><span class="token punctuation">(</span>Alpha<span class="token punctuation">)</span><span class="token punctuation">:</span>

   <span class="token keyword">def</span> <span class="token function">rebalance</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span><span class="token punctuation">:</span>
       self<span class="token punctuation">.</span>create_order<span class="token punctuation">(</span>
           <span class="token string">'ETH/USD'</span><span class="token punctuation">,</span>
           <span class="token number">1</span><span class="token punctuation">,</span>
           <span class="token string">'buy'</span><span class="token punctuation">,</span>
           route<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">)</span>
           
   <span class="token keyword">return</span>
</code></pre>
<p><code>self.create_order</code> can create market or limit orders. To create a limit order, specify a <code>limit_price</code>. If you want to immediately send the order to the broker pass <code>route=True</code>. Otherwise, it will wait to send orders until you call <code>self.route()</code>. Calling <code>self.route()</code> this way allows you to send all of your orders in the current rebalance to the broker at the same time if you’d like.<br>
<a id="create_order"></a></p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span> <span class="token function">create_order</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span>symbol<span class="token punctuation">,</span>qty<span class="token punctuation">,</span>side<span class="token punctuation">,</span>limit_price <span class="token operator">=</span> <span class="token boolean">None</span><span class="token punctuation">,</span>custom_id <span class="token operator">=</span> <span class="token boolean">None</span><span class="token punctuation">,</span>
	route <span class="token operator">=</span> <span class="token boolean">False</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
	<span class="token triple-quoted-string string">"""	
	Parameters
	----------
	symbol : str
	    Symbol for the security you want to trade. (IE. 'ETH/USD')
	qty :  float
	    Amount you want to trade. Must be positive.
	side : str ("buy" or "sell")
	    Side of trade. Must be "buy" or "sell".
	limit_price : float, optional
	    Price for a limit order.
	custom_id : str or float, optional
	    Custom internal identifier you want to attach to this order
	route : bool, optional
	    If True, immediately sends order to broker. 
	    If False, will wait until you call self.route()

	Returns
	-------
	None
	"""</span>
</code></pre>
<p>Another pre-built function for creating orders is <code>self.trade_to_target</code>. You can call this within the <code>rebalance</code> function of your strategy like this:</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">class</span> <span class="token class-name">ETHDCA</span><span class="token punctuation">(</span>Alpha<span class="token punctuation">)</span><span class="token punctuation">:</span>

    <span class="token keyword">def</span> <span class="token function">rebalance</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span><span class="token punctuation">:</span>
        target <span class="token operator">=</span> <span class="token punctuation">{</span><span class="token string">'ETH'</span><span class="token punctuation">:</span> <span class="token number">0.1</span><span class="token punctuation">,</span> <span class="token string">'BTC'</span><span class="token punctuation">:</span><span class="token number">0.01</span><span class="token punctuation">}</span>
        self<span class="token punctuation">.</span>trade_to_target<span class="token punctuation">(</span>target<span class="token punctuation">)</span>
</code></pre>
<p>This function creates market orders that  rebalance your positions to target positions specified in <code>target</code>. For instance, if you specify <code>target = {'ETH': 0.1}</code>, and your current ETH position is 0.2, it will create a market order to sell 0.1 ETH. This is useful for liquidity-taking strategies that best thought of as generating target positions rather than trades.<br>
<a id="trade_to_target"></a></p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span> <span class="token function">trade_to_target</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> target<span class="token punctuation">,</span> quote <span class="token operator">=</span> <span class="token string">'USD'</span><span class="token punctuation">,</span>route <span class="token operator">=</span> <span class="token boolean">False</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
	<span class="token triple-quoted-string string">"""
	Parameters
	----------
	target : dict
	    Dictionary with securities as keys and target positions as values.
	    Ex: {'ETH': 0.1 , 'BTC': 0.01}
	quote :  str, optional
	    Quote currency to use to trade to target.
	route : bool, optional
	    If True, immediately sends order to broker. 
	    If False, will wait until you call self.route()

	Returns
	-------
	None
	"""</span>
</code></pre>
<h3 id="canceling-orders">Canceling Orders</h3>
<p>The <code>Alpha</code> class has 2 pre-built functions for canceling orders:</p>
<ol>
<li><code>self.cancel_all_orders</code></li>
<li><code>self.cancel_order</code></li>
</ol>
<p>To cancel all of your strategy’s open orders, use <code>cancel_all_orders</code>.<br>
<a id="cancel_all_orders"></a></p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span> <span class="token function">cancel_all_orders</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span><span class="token punctuation">:</span>
    <span class="token triple-quoted-string string">"""
    Cancel all open orders
	
    Returns
    -------
    None
    """</span>
</code></pre>
<p>For instance, below code would cancel all outstanding orders on each rebalance to generate new trades on a clean slate.</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">class</span> <span class="token class-name">ETHDCA</span><span class="token punctuation">(</span>Alpha<span class="token punctuation">)</span><span class="token punctuation">:</span>
        
    <span class="token keyword">def</span> <span class="token function">rebalance</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span><span class="token punctuation">:</span>
        self<span class="token punctuation">.</span>cancel_all_orders<span class="token punctuation">(</span><span class="token punctuation">)</span>
        <span class="token triple-quoted-string string">"""
        More code to generate new trades.
        """</span>
</code></pre>
<p>To cancel a specific order, you can instead use <code>cancel_order</code> and pass in the <code>custom_id</code> of the order you’d like to cancel. Note that <code>custom_id</code> must have been set by you when you called <code>self.create_order</code>.<br>
<a id="cancel_order"></a></p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span> <span class="token function">cancel_order</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> custom_id<span class="token punctuation">)</span><span class="token punctuation">:</span>
    <span class="token triple-quoted-string string">"""
    Cancel order(s) specified by custom_id. 
    custom_id is attached to an order through self.create_order. 
    If you attached custom_id to more than 1 order, 
    all of them will be canceled.
  	
    Parameters
    ----------
    custom_id : str or float
        Custom identifier of order(s) to be canceled

    Returns
    -------
    None
    """</span>
</code></pre>
<h3 id="get-open-orders">Get Open Orders</h3>
<p>You can get your current open orders by calling the pre-built <code>self.get_open_orders</code> function of <code>Alpha</code>.</p>
<p>This will return a list of order objects containing the information of your open orders. You can pull relevant information from the order objects like this: <code>order.symbol, order.qty, order.side, order.limit_price, order.custom_id, order.qty_filled, order.avg_px</code></p>
<p>Below is an example where we act depending on whether we have outstanding orders for ‘ETH/USD’.</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">class</span> <span class="token class-name">OpenOrders</span><span class="token punctuation">(</span>Alpha<span class="token punctuation">)</span><span class="token punctuation">:</span>
    
    <span class="token keyword">def</span> <span class="token function">rebalance</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span><span class="token punctuation">:</span>
		    
        open_orders <span class="token operator">=</span> self<span class="token punctuation">.</span>get_open_orders<span class="token punctuation">(</span><span class="token punctuation">)</span>
        ethusd <span class="token operator">=</span> <span class="token punctuation">[</span>order <span class="token keyword">for</span> order <span class="token keyword">in</span> open_orders <span class="token keyword">if</span> order<span class="token punctuation">.</span>symbol <span class="token operator">==</span> <span class="token string">'ETH/USD'</span><span class="token punctuation">]</span>
        <span class="token keyword">if</span> <span class="token builtin">len</span><span class="token punctuation">(</span>ethusd<span class="token punctuation">)</span><span class="token punctuation">:</span>
            <span class="token triple-quoted-string string">"""
            do something
            """</span>
        <span class="token keyword">else</span><span class="token punctuation">:</span>
            <span class="token triple-quoted-string string">"""
            do something else
            """</span>
</code></pre>
<h3 id="react-to-finished-orders">React to Finished Orders</h3>
<p>You can “react” automatically to finished orders. This is more useful for “market-making” or “liquidity-providing” strategies.</p>
<p>Once an order is finished, the <code>on_finished_order</code> method of your strategy is always automatically called. So to react to an order, simply fill out this function. If you don’t fill it out, nothing will happen when an order finishes.</p>
<p>Below is an example for a “grid” strategy". Note that an <code>order</code> argument must always be set for <code>on_finished_order</code> because an <code>order</code> object containing all the finished order’s information will always be passed to <code>on_finished_order</code>. You can pull relevant order information from it such as: <code>order.symbol, order.qty, order.side, order.limit_price, order.custom_id, order.qty_filled, order.avg_px</code>.</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">class</span> <span class="token class-name">GRID</span><span class="token punctuation">(</span>Alpha<span class="token punctuation">)</span><span class="token punctuation">:</span>

    <span class="token keyword">def</span> <span class="token function">on_finished_order</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> order<span class="token punctuation">)</span><span class="token punctuation">:</span>
        side <span class="token operator">=</span> <span class="token string">'buy'</span> <span class="token keyword">if</span> order<span class="token punctuation">.</span>side<span class="token operator">==</span><span class="token string">'sell'</span> <span class="token keyword">else</span> <span class="token string">'sell'</span>
        
        <span class="token keyword">if</span> side <span class="token operator">==</span> <span class="token string">'buy'</span><span class="token punctuation">:</span>
            price <span class="token operator">=</span> order<span class="token punctuation">.</span>limit_price <span class="token operator">-</span> <span class="token number">5</span>
        <span class="token keyword">else</span><span class="token punctuation">:</span>
            price <span class="token operator">=</span> order<span class="token punctuation">.</span>limit_price <span class="token operator">+</span> <span class="token number">5</span>
            
        self<span class="token punctuation">.</span>create_order<span class="token punctuation">(</span>
            order<span class="token punctuation">.</span>symbol<span class="token punctuation">,</span>
            order<span class="token punctuation">.</span>qty<span class="token punctuation">,</span>
            side<span class="token punctuation">,</span>
            price<span class="token punctuation">,</span>
            route<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">)</span>
            
    <span class="token keyword">def</span> <span class="token function">rebalance</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span>order<span class="token punctuation">)</span><span class="token punctuation">:</span>
        <span class="token triple-quoted-string string">"""
        logic to create initial order
        """</span>
</code></pre>
<p>This very basic “grid” strategy  flips any filled limit orders. If you just sold a security, it will set a limit order to buy at $5 lower. If you just bought, it will plan to sell at $5 higher. The strategy is initialized in <code>rebalance</code> by the user with logic for creating an initial limit order. In this case, <code>rebalance</code> is used for initialization and most of the action occurs in <code>on_finished_order</code>.</p>
<p><a id="getpricedata"></a></p>
<h2 id="pricevolume-data">Price/Volume Data</h2>
<p>There are two primary <code>Alpha</code> functions for grabbing price-volume data.</p>
<ol>
<li><code>self.get_current_price</code></li>
<li><code>self.get_lastn_bars</code></li>
</ol>
<p>Currently, live trading price data is pulled from Kraken as that’s where live trading is done. Backtesting price data is pulled from Binance since it typically has longer price histories.</p>
<p>In the below example we use <code>get_current_price</code> to figure out the amount of ETH/USD to buy to obtain $100 of ETH.</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">class</span> <span class="token class-name">ETHDCA</span><span class="token punctuation">(</span>Alpha<span class="token punctuation">)</span><span class="token punctuation">:</span>

    <span class="token keyword">def</span> <span class="token function">rebalance</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span><span class="token punctuation">:</span>
        px <span class="token operator">=</span> self<span class="token punctuation">.</span>get_current_price<span class="token punctuation">(</span><span class="token string">'ETH/USD'</span><span class="token punctuation">)</span>
        amt <span class="token operator">=</span> <span class="token number">100</span> <span class="token operator">/</span> px
        self<span class="token punctuation">.</span>create_order<span class="token punctuation">(</span>
            <span class="token string">'ETH/USD'</span><span class="token punctuation">,</span>
            amt<span class="token punctuation">,</span>
            <span class="token string">'buy'</span><span class="token punctuation">,</span>
            route<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">)</span>
            
</code></pre>
<p>Here’s the simple function definition:<br>
<a id="get_current_price"></a></p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span> <span class="token function">get_current_price</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> symbol<span class="token punctuation">)</span><span class="token punctuation">:</span>
	<span class="token triple-quoted-string string">"""
	Parameters
	----------
	symbol : str
	    Security symbol (IE. 'ETH/USD')

	Returns
	-------
	float
	    Last trade price
	"""</span>
</code></pre>
<p>The <code>get_current_price</code> function can be handy for things like finalizing target quantities for market orders (as above), or for knowing where to set limit prices. Alpha signal generation, however, will more often be done using <code>get_lastn_bars</code> which provides much richer price data.</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">class</span> <span class="token class-name">ETHDCA</span><span class="token punctuation">(</span>Alpha<span class="token punctuation">)</span><span class="token punctuation">:</span>

    <span class="token keyword">def</span> <span class="token function">rebalance</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span><span class="token punctuation">:</span>
        bars <span class="token operator">=</span> self<span class="token punctuation">.</span>get_lastn_bars<span class="token punctuation">(</span><span class="token string">'ETH/USD'</span><span class="token punctuation">,</span> <span class="token number">5</span><span class="token punctuation">,</span> <span class="token string">'1h'</span><span class="token punctuation">)</span>
        <span class="token triple-quoted-string string">"""
        signal generation / order creation using bars
        """</span>
</code></pre>
<p>The above code will grab the last 5 1-hour bars for ‘ETH/USD’. And here’s the function details:<br>
<a id="get_lastn_bars"></a></p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span> <span class="token function">get_lastn_bars</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> symbol<span class="token punctuation">,</span> n<span class="token punctuation">,</span> freq<span class="token punctuation">,</span> lag<span class="token operator">=</span><span class="token number">0</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
	<span class="token triple-quoted-string string">"""
	Parameters
	----------
	symbol : str
	    Security symbol (IE. 'ETH/USD')
	n :  int
	    Number of bars to pull. In live trading, you can
	    only look back 720 bars.
	freq : {'1m', '5m','15m', '30m', '1h', '4h', '1d'}
	    Frequency (size) of the bars.
	    1m = 1 minute, 5m = 5 minutes, 15m = 15 minutes,
	    30m = 30 minutes, 1h = 1 hour, 4h = 4 hours,
	    1d = 1 day
	lag : int, optional
	    Whether to lag the bars. Using lag=2 with n=5 
	    is the same using lag=0, n=7 and dropping the 
	    most recent two bars returned.

	Returns
	-------
	DataFrame 
	columns = ['open','high','low','close','volume']
	index = Bar open times
	"""</span>
</code></pre>
<h2 id="strategy-parameters-and-preparation">Strategy Parameters and Preparation</h2>
<p>You can set custom strategy parameters or otherwise prepare your strategy for trading by filling out the <code>prepare</code> function. This function gets called <em>once</em> before the first rebalance when you launch the backtester or live trader. You pass your custom parameters through when you call <a href="#run-backtest">run_backtest</a> or <a href="#run-live">run_live</a></p>
<p>Here is pseudo-code for how <code>prepare</code> gets called in the background:</p>
<pre class=" language-python"><code class="prism  language-python">prepare<span class="token punctuation">(</span><span class="token punctuation">)</span>

<span class="token keyword">while</span> is_trading<span class="token punctuation">:</span>
    rebalance<span class="token punctuation">(</span><span class="token punctuation">)</span>
    sleep<span class="token punctuation">(</span>freq<span class="token punctuation">)</span>
</code></pre>
<p><code>prepare</code> is entirely optional. Not filling out <code>prepare</code> will simply do nothing.</p>
<p>Below example shows a simple HODL strategy where you can specify the symbol and amount to HODL. We then run a backtest for HODLing 2 ETH since 2021-01-01.</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">from</span> twsq<span class="token punctuation">.</span>alpha <span class="token keyword">import</span> Alpha

<span class="token keyword">class</span> <span class="token class-name">HODL</span><span class="token punctuation">(</span>Alpha<span class="token punctuation">)</span><span class="token punctuation">:</span>

    <span class="token keyword">def</span> <span class="token function">prepare</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span>symbol<span class="token punctuation">,</span> amount<span class="token punctuation">)</span><span class="token punctuation">:</span>
        self<span class="token punctuation">.</span>symbol <span class="token operator">=</span> symbol
        self<span class="token punctuation">.</span>qty <span class="token operator">=</span> qty
        self<span class="token punctuation">.</span>first_rebalance <span class="token operator">=</span> <span class="token boolean">True</span>
        
    <span class="token keyword">def</span> <span class="token function">rebalance</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span><span class="token punctuation">:</span>
        <span class="token keyword">if</span> self<span class="token punctuation">.</span>first_rebalance<span class="token punctuation">:</span>
            self<span class="token punctuation">.</span>create_order<span class="token punctuation">(</span>
                self<span class="token punctuation">.</span>symbol<span class="token punctuation">,</span>
                self<span class="token punctuation">.</span>qty<span class="token punctuation">,</span>
                <span class="token string">'buy'</span><span class="token punctuation">,</span>
                route<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">)</span>
            self<span class="token punctuation">.</span>first_rebalance <span class="token operator">=</span> <span class="token boolean">False</span>  
    <span class="token keyword">return</span>

HODL<span class="token punctuation">.</span>run_backtest<span class="token punctuation">(</span>HODL<span class="token punctuation">,</span> start_ts<span class="token operator">=</span><span class="token string">'20210101'</span><span class="token punctuation">,</span>symbol <span class="token operator">=</span> <span class="token string">'ETH/USD'</span><span class="token punctuation">,</span> qty<span class="token operator">=</span><span class="token number">2</span><span class="token punctuation">)</span>
</code></pre>
<p>Note how, although <code>first_rebalance</code> is not a parameter you can set, you still have to initialize it in <code>prepare</code> to use it correctly.</p>
<p>Other potential operations to include in prepare could be time-consuming initial data loads or manipulations. For instance, if you wanted to train or initialize a machine-learning algorithm prior to trading.</p>
<h2 id="get-positions">Get Positions</h2>
<p>You can pull your strategy’s current positions using pre-built <code>Alpha</code> function <code>get_pos</code>, which returns your positions as a dictionary:</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span> <span class="token function">get_pos</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span><span class="token punctuation">:</span>
    <span class="token triple-quoted-string string">"""
    Returns positions of your strategy as a dictionary
    
    Returns
    -------
    Dict
    keys = security name (IE. "ETH")
    values = amount held
    """</span>
</code></pre>
<p>Note that the returned positions are for your strategy only. It excludes any other positions you have with your broker. Moreover, the positions returned are essentially cumulative trade values. So if you bought $1000 of ETH/USD, your position would have the $1000 equivalent of ETH, but also a minus $ (1000 + x) USD position. Where the x represents comissions.</p>
<p>Below is an example using <code>get_pos</code>. It buys $1000 of BTC and $1000 of ETH on the first rebalance. Subsequently, it calls <code>get_pos</code> and if the $ amounts in BTC and ETH are not equal, it trades BTC/ETH to keep them at equal weights.</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">class</span> <span class="token class-name">EqualWeights</span><span class="token punctuation">(</span>Alpha<span class="token punctuation">)</span><span class="token punctuation">:</span>
    
    <span class="token keyword">def</span> <span class="token function">prepare</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span><span class="token punctuation">:</span>
        self<span class="token punctuation">.</span>first_rebalance <span class="token operator">=</span> <span class="token boolean">True</span>
        
    <span class="token keyword">def</span> <span class="token function">rebalance</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span><span class="token punctuation">:</span>
    
        <span class="token keyword">if</span> self<span class="token punctuation">.</span>first_rebalance<span class="token punctuation">:</span>
            eth_qty <span class="token operator">=</span> <span class="token number">1000</span><span class="token punctuation">.</span><span class="token operator">/</span>self<span class="token punctuation">.</span>get_current_price<span class="token punctuation">(</span><span class="token string">'ETH/USD'</span><span class="token punctuation">)</span>
            btc_qty <span class="token operator">=</span> <span class="token number">1000</span><span class="token punctuation">.</span><span class="token operator">/</span>self<span class="token punctuation">.</span>get_current_price<span class="token punctuation">(</span><span class="token string">'BTC/USD'</span><span class="token punctuation">)</span>
                
            self<span class="token punctuation">.</span>create_order<span class="token punctuation">(</span>
                <span class="token string">'ETH/USD'</span><span class="token punctuation">,</span>
                eth_qty<span class="token punctuation">,</span>
                <span class="token string">'buy'</span><span class="token punctuation">,</span>
                route<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">)</span>
             
             self<span class="token punctuation">.</span>create_order<span class="token punctuation">(</span>
                <span class="token string">'BTC/USD'</span><span class="token punctuation">,</span>
                btc_qty<span class="token punctuation">,</span>
                <span class="token string">'buy'</span><span class="token punctuation">,</span>
                route<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">)</span>
            
            self<span class="token punctuation">.</span>first_rebalance <span class="token operator">=</span> <span class="token boolean">False</span>

        <span class="token keyword">else</span><span class="token punctuation">:</span>

            pos <span class="token operator">=</span> self<span class="token punctuation">.</span>get_pos<span class="token punctuation">(</span><span class="token punctuation">)</span>

            eth_pos <span class="token operator">=</span> pos<span class="token punctuation">.</span>get<span class="token punctuation">(</span><span class="token string">'ETH'</span><span class="token punctuation">,</span><span class="token number">0</span><span class="token punctuation">)</span>
            px_eth <span class="token operator">=</span> self<span class="token punctuation">.</span>get_current_price<span class="token punctuation">(</span><span class="token string">'ETH/USD'</span><span class="token punctuation">)</span>
            eth_dollars <span class="token operator">=</span> eth_pos<span class="token operator">*</span>px_eth

            btc_pos <span class="token operator">=</span> pos<span class="token punctuation">.</span>get<span class="token punctuation">(</span><span class="token string">'BTC'</span><span class="token punctuation">,</span><span class="token number">0</span><span class="token punctuation">)</span>
            px_btc <span class="token operator">=</span> self<span class="token punctuation">.</span>get_current_price<span class="token punctuation">(</span><span class="token string">'BTC/USD'</span><span class="token punctuation">)</span>
            btc_dollars <span class="token operator">=</span> btc_pos<span class="token operator">*</span>px_btc

            qty <span class="token operator">=</span> <span class="token number">0.5</span> <span class="token operator">*</span> <span class="token builtin">abs</span><span class="token punctuation">(</span>btc_dollars <span class="token operator">-</span> eth_dollars<span class="token punctuation">)</span> <span class="token operator">/</span> px_btc
            side <span class="token operator">=</span> <span class="token string">'buy'</span> <span class="token keyword">if</span> btc_dollars <span class="token operator">&lt;</span> eth_dollars <span class="token keyword">else</span> <span class="token string">'sell'</span>

            self<span class="token punctuation">.</span>create_order<span class="token punctuation">(</span>
                <span class="token string">'BTC/ETH'</span><span class="token punctuation">,</span>
                qty<span class="token punctuation">,</span>
                side<span class="token punctuation">,</span>
                route<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">)</span>
</code></pre>
<h2 id="custom-data">Custom Data</h2>
<p>Currently integrated data for strategies is solely price-volume data. This should give you plenty to play with as price-volume strategies are some of the most powerful and crypto markets are some of the least efficient.</p>
<p>If you’d really like to incorporate additional data, you should be able to by coding your strategy appropriately.</p>
<p>While it will differ by dataset, below is <em>rough</em> pseudo-code for doing this.</p>
<pre class=" language-python"><code class="prism  language-python">
<span class="token keyword">class</span> <span class="token class-name">CustomData</span><span class="token punctuation">:</span>
    
    <span class="token keyword">def</span> <span class="token function">get_data</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span>ts<span class="token punctuation">)</span><span class="token punctuation">:</span>
        <span class="token triple-quoted-string string">"""
        Get custom data at time ts using API or downloaded data
        """</span>

<span class="token keyword">class</span> <span class="token class-name">CustomDataExample</span><span class="token punctuation">(</span>Alpha<span class="token punctuation">)</span><span class="token punctuation">:</span>

    <span class="token keyword">def</span> <span class="token function">prepare</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span>customdata<span class="token operator">=</span><span class="token boolean">None</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
        self<span class="token punctuation">.</span>customdata <span class="token operator">=</span> customdata
        
    <span class="token keyword">def</span> <span class="token function">rebalance</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span><span class="token punctuation">:</span>
        data_now <span class="token operator">=</span> self<span class="token punctuation">.</span>customdata<span class="token punctuation">.</span>get_data<span class="token punctuation">(</span>self<span class="token punctuation">.</span>ts<span class="token punctuation">)</span>
        <span class="token triple-quoted-string string">"""
        Manipulations using data_now to generate trades
        """</span>
       
CustomDataExample<span class="token punctuation">.</span>run_backtest<span class="token punctuation">(</span>start_ts<span class="token operator">=</span><span class="token string">'20210101'</span><span class="token punctuation">,</span>customdata<span class="token operator">=</span>CustomData<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
</code></pre>
<p>In short, you could create a class <code>CustomData</code> with a <code>get_data</code> function that can pull your data through, for instance, an API. <code>get_data</code> takes as input the current timestamp <code>ts</code> through the <code>self.ts</code> property of <code>Alpha</code> (<code>self.ts</code> returns the current UTC timestamp in live trading or the historical UTC timestamp of the current iteration in backtesting). You should then pass this <code>CustomData</code> class into your strategy through the <code>prepare</code> method (introduced <a href="#strategy-parameters-and-preparation">here</a>).</p>
<p>Feel free to contact us for further help on incorporating additional datasets.</p>
<p>We are working on integrating new datasets as well, and will keep you posted.</p>
<p><a id="on_exit"></a></p>
<h2 id="exiting">Exiting</h2>
<p>You can fill out the optional function <code>on_exit</code> to tell your trader what to do when it exits or crashes. Below is an example using <code>on_exit</code> to automatically cancel any open orders when the trader exits.</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">class</span> <span class="token class-name">ExitExample</span><span class="token punctuation">(</span>Alpha<span class="token punctuation">)</span><span class="token punctuation">:</span>
     <span class="token keyword">def</span> <span class="token function">on_exit</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span><span class="token punctuation">:</span>
        self<span class="token punctuation">.</span>cancel_all_orders<span class="token punctuation">(</span><span class="token punctuation">)</span>
</code></pre>
<h1 id="backtesting">Backtesting</h1>
<h2 id="launching">Launching</h2>
<p>The example in the <a href="#backtest-intro">introduction</a>, copied below, shows we can run a backtest by using the <code>run_backtest</code> method:</p>
<pre class=" language-python"><code class="prism  language-python">ETHDCA<span class="token punctuation">.</span>run_backtest<span class="token punctuation">(</span>start_ts<span class="token operator">=</span><span class="token string">'20210101'</span><span class="token punctuation">)</span>
</code></pre>
<p>This method has several optional parameters:<br>
<a id="run-backtest"></a></p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span> <span class="token function">run_backtest</span><span class="token punctuation">(</span>cls<span class="token punctuation">,</span> start_ts <span class="token operator">=</span> <span class="token boolean">None</span><span class="token punctuation">,</span> end_ts <span class="token operator">=</span> <span class="token boolean">None</span><span class="token punctuation">,</span>
		freq<span class="token operator">=</span><span class="token string">'1h'</span><span class="token punctuation">,</span>name<span class="token operator">=</span><span class="token boolean">None</span><span class="token punctuation">,</span> taker_fee<span class="token operator">=</span><span class="token boolean">None</span><span class="token punctuation">,</span>maker_fee<span class="token operator">=</span><span class="token boolean">None</span><span class="token punctuation">,</span>slip<span class="token operator">=</span><span class="token boolean">None</span><span class="token punctuation">,</span>
		<span class="token operator">**</span>kwargs<span class="token punctuation">)</span><span class="token punctuation">:</span>
       <span class="token triple-quoted-string string">"""
       Run a backtest on Alpha

       Parameters
       ----------
       start_ts : str, datetime or pandas timestamp, optional
           start time of backtest. default is 365 days prior to end_ts
       end_ts : str, datetime or pandas timestamp, optional
           end time of backtest. default is current time.
       freq : {'1m', '5m','15m', '30m', '1h', '4h', '1d'}, optional
           Rebalance or iteration frequency. 
           IE '1m' will call self.rebalance every minute.
           1m = 1 minute, 5m = 5 minutes, 15m = 15 minutes,
           30m = 30 minutes, 1h = 1 hour, 4h = 4 hours,
           1d = 1 day
       name :  str, optional
           Name of strategy. Used in logging and saving down results in MyTWSQ. 
           Will default to the name of the Alpha class passed in.
       taker_fee : float, optional
           Commissions for market orders. Will default to Kraken's fee of 26 bps
           per dollar traded.
       maker_fee : float, optional
           Commissions for limit orders. Will default to Kraken's fee of 16 bps
           per dollar traded.
       slip : float, optional
           Slippage for market orders. Will default to 10 bps per dollar traded.

       Optional custom strategy parameters can be passed through **kwargs
    
       Returns
       -------
       Alpha
       Your instantiated strategy (Alpha sub-class)
       Use Alpha.pos_pnl to see backtest positions and pnl
       Use Alpha.orders to see all backtest orders 
       """</span>
</code></pre>
<p>The <code>freq</code> parameter determines how often the backtester calls <code>rebalance</code>.  Possible values are <code>{'1m', '5m','15m', '30m', '1h', '4h', '1d'}</code>.</p>
<p><code>name</code> is the name of the backtest, which is used for logging and for naming the folder to save down results in MyTWSQ. It defaults to the name of your alpha class (IE. <code>ETHDCA</code> in the above example).</p>
<p>You can also optionally specify trading costs using <code>taker_fee</code>, <code>maker_fee</code> and <code>slip</code>. These should be specified in cost per dollar traded. So to specify a <code>slip</code> of 5 bps per dollar traded you would pass 5e-4. The defaults are listed above.</p>
<p>Below is an example where we run two separate backtests for the <code>ETHDCA</code> example from the <a href="#create-a-strategy">introduction</a>. Once with transaction costs, which we name “Net”, and once without transaction, named “Gross”. We might run this if we wanted to see the impact fees had on our results.</p>
<pre class=" language-python"><code class="prism  language-python">net <span class="token operator">=</span> ETHDCA<span class="token punctuation">.</span>run_backtest<span class="token punctuation">(</span>start_ts<span class="token operator">=</span><span class="token string">'20210101'</span><span class="token punctuation">,</span>name <span class="token operator">=</span> <span class="token string">'Net'</span><span class="token punctuation">,</span>freq<span class="token operator">=</span><span class="token string">'1d'</span><span class="token punctuation">)</span>
gross <span class="token operator">=</span> ETHDCA<span class="token punctuation">.</span>run_backtest<span class="token punctuation">(</span>start_ts<span class="token operator">=</span><span class="token string">'20210101'</span><span class="token punctuation">,</span>name <span class="token operator">=</span> <span class="token string">'Gross'</span><span class="token punctuation">,</span>freq<span class="token operator">=</span><span class="token string">'1d'</span><span class="token punctuation">,</span> 
	taker_fee <span class="token operator">=</span> <span class="token number">0</span><span class="token punctuation">,</span> maker_fee <span class="token operator">=</span> <span class="token number">0</span><span class="token punctuation">,</span> slip <span class="token operator">=</span> <span class="token number">0</span><span class="token punctuation">)</span>
</code></pre>
<p>As discussed in <a href="#strategy-parameters-and-preparation">Strategy Parameters and Preparation</a>, <code>run_backtest</code> can also receive custom strategy parameters specified by filling out the <code>prepare</code> method.</p>
<p>The <code>run_backtest</code> method returns your instantiated strategy object, with which you can view backtest results. In the above example, for the “Net” backtest we could call <code>net.pos_pnl</code> to obtain backtest positions &amp; pnl and <code>net.orders</code> to obtain all backtest order information. These are identical to the csv files for the backtest saved down in <code>MyTWSQ</code> further discussed <a href="#backtest-output">here</a>.</p>
<h2 id="core-backtesting-logic">Core Backtesting Logic</h2>
<p>The key variable the backtester uses is the <code>freq</code> you pass in <code>run_backtest</code>. The backtester iterates through history at a frequency of <code>freq</code>. So, if you are using <code>freq=5m</code>, it will iterate through each 5 minute bar in the backtest date range.</p>
<p>Every 5m, it will call the <code>rebalance</code> function of your alpha. Any market orders created by <code>rebalance</code> are assumed to be filled immediately at the current price, with slippage. If slippage is 5bps per dollar traded, buys are filled at <code>(current price + (5e-4*current_price))</code>. Sells are filled at <code>(current price - (5e-4*current_price))</code></p>
<p>Any limit orders generated are checked every 5m if they are filled. Let’s say you generated a limit order using <code>rebalance</code> at 2021-03-01 15:00:00.</p>
<p>Then, at the next time step, 2021-03-01 15:05:00, the backtester checks if any limit orders should be filled, and fills and updates positions if necessary <em>before</em> <code>rebalance</code> is called again.</p>
<p>The logic for this is simple. If you have a sell limit order, and the high of the 5m bar (2021-03-01 15:00:00,  2021-03-01 15:05:00) is higher than your limit price, fill it. Likewise, for a buy limit order, if the low of the 5m bar is lower than your limit price, fill it. If any limit order is unfilled, the backtester will continue to check if it is filled or not in each subsequent iteration.</p>
<p>Below is pseudo-code for the core backtesting loop:</p>
<pre class=" language-python"><code class="prism  language-python">prepare<span class="token punctuation">(</span><span class="token punctuation">)</span>
ts <span class="token operator">=</span> start_ts
<span class="token keyword">while</span> ts <span class="token operator">&lt;</span> end_ts<span class="token punctuation">:</span>
    fill_limit_orders<span class="token punctuation">(</span><span class="token punctuation">)</span>
    rebalance<span class="token punctuation">(</span><span class="token punctuation">)</span>
    fill_market_orders<span class="token punctuation">(</span><span class="token punctuation">)</span>
    ts <span class="token operator">=</span> ts <span class="token operator">+</span> freq
</code></pre>
<h2 id="pnl-calculation">PnL Calculation</h2>
<p>Your strategy executes a series of trades. The backtest PnL is simply the aggregate PnL of those trades.</p>
<p>Importantly, there is no notion of “budget” or “equity capital” to bound it. So PnL can be infinitely negative in the backtest. In this sense, the backtest is indicative of how your strategy would run if you had infinite resources to fund it. You can then use this information to decide whether and how to fund it live.</p>
<p>Backtest PnL is stored in <code>MyTWSQ/alphas/name/backtest/pos_pnl.csv</code>.  See the <a href="#output">output</a> section below for further details.</p>
<h2 id="shorting">Shorting</h2>
<p>Shorting is currently permitted in the backtest. We are working on rolling it out for live trading as well.</p>
<p><a id="backtest-output"></a></p>
<h2 id="output">Output</h2>
<p>The backtest results (Trades, PnL, Positions, etc.) are saved in <code>MyTWSQ/alphas/name/backtest</code> (where name is optionally specified when you call <a href="#run-backtest">run_backtest</a>. Two csv files are saved:</p>
<ol>
<li>pos_pnl.csv</li>
<li>orders.csv</li>
</ol>
<p>pos_pnl.csv has the following columns:</p>
<ul>
<li><code>Date</code> -  Timestamp. Iterates through the backtest <code>start_ts</code> and <code>end_ts</code> at increments of size <code>freq</code>.</li>
<li>1 column for each asset representing the strategy’s amount held. This is the cumulative sum of the impact of all trades on the asset, including fees.</li>
<li><code>port_val</code> - Portfolio value in USD. Sum of all positions in USD terms.</li>
<li><code>pnl</code> - Profit or loss between the current <code>Date</code> and the prior <code>Date</code>. Note this is simply the difference in the <code>port_val</code> on those dates.</li>
</ul>
<p>We should note that <code>port_val</code> represents <em>net</em> portfolio value due to your strategy’s trades. It  is the sum of your strategy positions, <em>including USD</em>, in USD terms and will not include any positions or trades your strategy did not generate. In this sense, it is basically a <em>cumulative pnl</em>.</p>
<p>Example. You started out with $10,000 in your account. You bought ETH/USD for $1000. Your <code>port_val</code> would be roughly -$2.6. Your strategy consumed $1000 USD in exchange for $1000 ETH. So the “net” result of that is zero. However, you pay roughly $2.6 in commissions. This is more like a “long-short zero net investment” notion of portfolio value common in the Hedge Fund world. Your USD position after this would be -$1002.6 and your ETH position would be long the amount needed to be roughly $1000 in USD terms.</p>
<p>Another notion of portfolio value, which you could compute given the data, is the sum of the value of your non USD positions.</p>
<p>orders.csv stores all your orders. It has many mostly self-explanatory columns: <code>strategy, symbol, qty, side, sec_type, limit_price, type, custom_id, id, status, qty_filled, ntn_filled, avg_px, fee, start_ts, arrival_px, base, quote, end_ts</code>.</p>
<p>A few clarifications:</p>
<ul>
<li><code>id</code> column represents the identifier the <em>broker</em> attaches to your order. You can look up your order with your broker using this identifier.</li>
<li><code>qty</code> is the original amount you tried to trade.</li>
<li><code>qty_filled</code> is the quantity of your order that has been filled. For backtests, this will always be 0 or the full order quantity. Live trading may have partial fills.</li>
<li><code>avg_px</code> is the average price your order was executed at.</li>
<li><code>ntn_filled</code> is <code>qty_filled * avg_px</code></li>
<li><code>arrival_px</code> is the arrival price of your order. Useful for measuring slippage.</li>
</ul>
<h1 id="live-trading-1">Live Trading</h1>
<h2 id="configure-settings-file">Configure Settings File</h2>
<p>Once you’ve set your <code>TWSQROOT</code> directory (instructions <a href="#set-the-twsqroot-directory">here</a>), create the <code>MyTWSQ</code> folder there (if it’s not there already).</p>
<p>Next, create a new file named <code>settings.yml</code> in your folder. You can do so by simply creating a new file using your editor (VS Code, Sublime, etc.) and naming it <code>settings.yml</code>. This file will contain user-specific credentials and settings.</p>
<p>For now, the key information to add is the API key and secret used for live trading. The library currently trades on Kraken, so please create an account there if you don’t already have one. <a href="https://support.kraken.com/hc/en-us/articles/360000919966-How-to-generate-an-API-key-pair-">Here</a> is a guide on how to obtain your API key and secret from Kraken once you have an account.</p>
<p>Add your key and secret to the <code>settings.yml</code> file under <code>exchange_apis</code> like so</p>
<pre class=" language-yml"><code class="prism  language-yml"><span class="token key atrule">exchange_apis</span><span class="token punctuation">:</span>
  <span class="token key atrule">Kraken</span><span class="token punctuation">:</span>
    <span class="token key atrule">key</span><span class="token punctuation">:</span> yourkeyhere
    <span class="token key atrule">secret</span><span class="token punctuation">:</span> yoursecrethere
</code></pre>
<h2 id="launching-1">Launching</h2>
<p>The example in the introduction, copied below, shows we can run the live trader by using the <code>run_live</code> method:</p>
<pre class=" language-python"><code class="prism  language-python">ETHDCA<span class="token punctuation">.</span>run_live<span class="token punctuation">(</span><span class="token punctuation">)</span>
</code></pre>
<p>This method is defined as such:<br>
<a id="run-live"></a></p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span> <span class="token function">run_live</span><span class="token punctuation">(</span>cls<span class="token punctuation">,</span> freq<span class="token operator">=</span><span class="token string">'1h'</span><span class="token punctuation">,</span>name<span class="token operator">=</span><span class="token boolean">None</span><span class="token punctuation">,</span> <span class="token operator">**</span>kwargs<span class="token punctuation">)</span><span class="token punctuation">:</span>

    <span class="token triple-quoted-string string">"""
    Run live trading for your Alpha
    Parameters
    ----------
    freq : float or {'1m', '5m','15m', '30m', '1h', '4h', '1d'}, optional
        Rebalance or iteration frequency. 

        You can specifiy the number of seconds between each rebalance by passing
        a float. In this case, you rebalance immediately and every freq seconds
        thereafter.

        Or you can use one of the preset rebalance frequencies:

        1m = 1 minute, 5m = 5 minutes, 15m = 15 minutes,
        30m = 30 minutes, 1h = 1 hour, 4h = 4 hours,
        1d = 1 day.

        If you use these, you only rebalance on bar end times. So if it is 3:15
        PM and you call this function with 30m you won't rebalance until 3:30
        PM. Next you'll rebalance at 4:00PM, 4:30 PM, etc

    name :  str, optional
        Name of strategy. Used in logging and for naming the folder to save 
        down results in MyTWSQ. Will default to the name of the Alpha class
        passed in.
    
    Optional custom strategy parameters can be passed through **kwargs
    """</span>
</code></pre>
<p>The <code>freq</code> parameter determines how often the live trader calls <code>rebalance</code>.</p>
<p>Possible preset frequencies are <code>{'1m', '5m','15m', '30m', '1h', '4h', '1d'}</code>. With these, rebalancing is done at bar starts times. So if you launch a session at 3:15 PM and your <code>freq</code> is 30m, you won’t rebalance until 3:30 PM. This is nice because it aligns with how the backtester rebalances.</p>
<p>You can instead specify the number of seconds between each rebalance by passing a float. In this case, you rebalance immediately and every <code>freq</code> seconds thereafter.</p>
<p>The <code>name</code> parameter is the name of the trading session, which is used for logging and for naming the folder to save down results in MyTWSQ. It defaults to the name of your alpha class (IE. <code>ETHDCA</code> in the above example).</p>
<h2 id="exchange-api-set-up">Exchange API Set Up</h2>
<p>The trader uses a combination of REST and WebSocket exchange APIs to obtain pricing data and manage orders. This (optional) section introduces when and why we use each.</p>
<p>Practically speaking, REST allows you to post requests to the exchange and receive responses, while WebSocket establishes a continuous connection with the exchange through which you can send and receive data.</p>
<p>Example. Say you wanted the last trade price of BTC/USD. Using REST, you would have to post a request to the exchange each time you wanted this. The exchange would then send you a response with the price. With a WebSocket connection, the exchange will automatically send you any updates to the last trade price in real time. Another Example. REST = checking your phone for new text messages manually. WebSocket = Enabling push notifications.</p>
<p>The advantages of a WebSocket  are:</p>
<ol>
<li>You get any data updates much faster and in real time. So you don’t have to worry about “checking” for updates by posting requests to the exchange.</li>
<li>Most exchanges actually have limits on the number of REST calls you can make per unit time. Kraken’s limits are <a href="https://support.kraken.com/hc/en-us/articles/206548367-What-are-the-API-rate-limits">here</a>.</li>
</ol>
<p>The advantages of REST are:</p>
<ol>
<li>Simpler to code.</li>
<li>WebSocket API  connections can be interrupted for a variety of reasons. So you need to also implement reconnecting logic appropriately.</li>
</ol>
<p>Which do we use? WebSockets are used predominantly for receiving pricing data updates in real time (last trade price or bars) and for receiving updates to outstanding orders (is it finished? how much has been filled? etc.). REST is used for submitting orders.</p>
<p>You’d likely want to use WebSocket for something like receiving updates to your outstanding orders. That way you don’t have to keep checking when/if your order has been filled, which would eat up your rate limit. Moreover, you can obtain any updates in real time so you can react quickly if necessary. Similarly, obtaining pricing updates with WebSocket is nice because it’s faster and again preserves your rate limit.</p>
<p>Submitting orders using REST works because there’s no immediate response or data you need afterwards. The point is simply to send something to the exchange. The downside of course is that you may hit <a href="https://support.kraken.com/hc/en-us/articles/206548367-What-are-the-API-rate-limits">rate limits</a> if you send too many orders. In most cases, however, we should be okay, and we will migrate order submission to WebSocket as well in the future.</p>
<h2 id="exiting-1">Exiting</h2>
<p>If you’d like to exit your trader, you can simply use a keyboard interruption CTRL+C. You can fill out the optional function <a href="#on_exit">on_exit</a> to have your trader perform any final tasks before exiting.</p>
<p>Do not close your terminal to exit as that will not allow the trader to save down final results and call the <a href="#on_exit">on_exit</a> function.</p>
<h2 id="output-1">Output</h2>
<p>The live trading results are saved in <code>MyTWSQ/alphas/name/live_trading/broker/</code> (where name is optionally specified when you call <code>run_live</code>, and broker is by default “Kraken”).  Saved info includes:</p>
<ol>
<li>logs</li>
<li>pos_pnl.csv</li>
<li>orders.csv</li>
</ol>
<p>Timestamped log files with all live logging activity are stored in the <code>logs</code> sub directory.</p>
<p>The two csv files mirror the backtesting output, and are largely similar to what we discussed in that section. There are two relevant differences.</p>
<p>One is that pos_pnl.csv file has an entry every 5 minutes (by default) and when an order is filled, rather than on each rebalance as for backtesting. This simply ensures we always have a sufficiently granular read on pnl in live trading.</p>
<p>Second, these files maintain continuity across trading sessions for the same alpha. Let’s say you halted a trading session. The next time you start a trading session with an alpha of the same name, it will <em>append</em> to the pos_pnl.csv and orders.csv files generated initially. This way you always maintain a complete picture of your strategy in one spot.</p>
<h2 id="position-tracking">Position Tracking</h2>
<p>Your strategy positions are maintained across trading sessions.</p>
<p>Say you started a trading session and bought 1 ETH. Then you decide to halt the session to modify your strategy, or your computer crashes. If you restart a trading session and pass the same <code>name</code>, it will load the 1 ETH as your initial position using data stored in the pos_pnl.csv file mentioned above.</p>
<p>So if you call <code>get_pos</code> you will see a position of 1 ETH.</p>
<p>It’s important to maintain this kind of continuity across trading sessions so you don’t have strategy positions in limbo and can trade to target positions correctly.</p>
<h2 id="telegram-logging">Telegram Logging</h2>
<p>It’s important to have mobile notifications or alerts on the status of your live trader since you’ll likely step away from your computer many times while it’s running.</p>
<p>Here we discuss how you can set up <a href="https://telegram.org/">telegram messenger</a> to send the trader’s logging directly to your phone. We will be using the <a href="https://github.com/sashgorokhov/python-telegram-handler">python-telegram-handler</a> package. While this is optional, it is recommended.</p>
<p>First download the telegram app on your phone.</p>
<p>Next you will need to create a telegram chat bot (full instructions <a href="https://core.telegram.org/bots#6-botfather">here</a>). Initiate a telegram conversation with <a href="https://t.me/botfather">BotFather</a> and send him a message that says <code>/start</code>. Follow the instructions to set up your bot and obtain your <code>token</code>, a string that looks something like <code>110201543:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw</code>.</p>
<p>Your new bot should appear in your telegram app. Send your new bot a message (anything) to first initiate it. Now run this command <code>python -m telegram_handler &lt;your token here&gt;</code>  in your terminal to obtain a <code>chat_id</code> number. This command won’t work unless you’ve sent your bot an initial message.</p>
<p>Add your <code>token</code> and <code>chat_id</code> to the <code>settings.yml</code> file (discussed above) like so:</p>
<pre class=" language-yml"><code class="prism  language-yml"><span class="token key atrule">telegram</span><span class="token punctuation">:</span>
    <span class="token key atrule">token</span><span class="token punctuation">:</span> 110201543<span class="token punctuation">:</span>AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw
    <span class="token key atrule">chat_id</span><span class="token punctuation">:</span> <span class="token number">0123456789</span>
</code></pre>
<p>Now when you run the trader, all the logging will also be sent to your telegram bot and you can view it on your phone.</p>
<h1 id="easy-price-data">Easy Price Data</h1>
<p>Here we introduce an easy method for pulling price data from Binance through this library.</p>
<p>Note that this is not how you should pull data when coding your strategy in this library. See <a href="#getpricedata">above</a> for how to do this. This method of pulling data is if you want to do preliminary research before using the library’s backtesting and live trading engine</p>
<p>Here’s how to do it:</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">from</span> twsq<span class="token punctuation">.</span>data <span class="token keyword">import</span> BinancePrices
px <span class="token operator">=</span> BinancePrices<span class="token punctuation">(</span><span class="token punctuation">)</span>
btc_px <span class="token operator">=</span> px<span class="token punctuation">.</span>get_bars<span class="token punctuation">(</span><span class="token string">'BTC/USD'</span><span class="token punctuation">,</span> <span class="token string">'4h'</span><span class="token punctuation">)</span>
</code></pre>
<p>The main benefit is that the library downloads and caches the price data such that you don’t have to keep downloading it each time. This can be helpful for slow downloads for granular (e.g. 1 min) bars. Here’s the function definition:</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span> <span class="token function">get_bars</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> symbol<span class="token punctuation">,</span> freq <span class="token operator">=</span> <span class="token string">'1d'</span><span class="token punctuation">,</span> start_ts <span class="token operator">=</span> <span class="token boolean">None</span><span class="token punctuation">,</span> end_ts <span class="token operator">=</span> <span class="token boolean">None</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
    <span class="token triple-quoted-string string">"""
    Parameters
    ----------
    symbol : str
        Symbol to pull bars for
    freq : str {'1m', '5m','15m', '30m', '1h', '4h', '1d'}
        Frequency (size) of bars. 1m = 1 minute, 1h = 1 hour,
        1d = 1 day, etc. 
    start_ts : str, datetime or pandas timestamp, optional
        start time of bar history. defaults to earliest available.
    end_ts : str, datetime or pandas timestamp, optional
        end time of bar history. default is most recent bar

    Returns
    -------
    DataFrame 
    columns = ['open','high','low','close','volume']
    index = Bar open times
    """</span>
</code></pre>
<p>On the first call, the function will download all bars of size <code>freq</code> for <code>symbol</code> and cache it as a pickle file. On each subsequent call, if <code>end_ts</code> is contained within the data in the pickle file, the function simply reads from the pickle. If <code>end_ts</code> is beyond what has been downloaded, the incremental data will be downloaded, cached and then returned.</p>
<p>Pickle files are stored in MyTWSQ/data/Binance.</p>

    </div>
  </div>
</body>

</html>
