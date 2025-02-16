"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import Link from "next/link"
import { Search, ChevronLeft, Copy } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"

function StarryBackground() {
  const [stars, setStars] = useState<Array<{ id: number; x: number; y: number; size: number; delay: number }>>([])

  useEffect(() => {
    const newStars = Array.from({ length: 100 }, (_, i) => ({
      id: i,
      x: Math.random() * 100,
      y: Math.random() * 100,
      size: Math.random() * 0.1 + 0.05,
      delay: Math.random() * 3,
    }))
    setStars(newStars)
  }, [])

  return (
    <div className="fixed inset-0 pointer-events-none opacity-20">
      <svg className="w-full h-full" viewBox="0 0 100 100">
        {stars.map((star) => (
          <motion.circle
            key={star.id}
            cx={star.x}
            cy={star.y}
            r={star.size}
            fill="white"
            initial={{ opacity: 0.1 }}
            animate={{
              opacity: [0.2, 0.8, 0.2],
              scale: [1, 1.2, 1],
            }}
            transition={{
              duration: 2 + star.delay,
              repeat: Number.POSITIVE_INFINITY,
              ease: "easeInOut",
            }}
          />
        ))}
      </svg>
    </div>
  )
}

const sections = {
  "GET STARTED": [
    { title: "Overview", href: "/docs/overview" },
    { title: "Quickstart", href: "/docs/quickstart" },
    { title: "Installation", href: "/docs/installation" },
  ],
  "CORE CONCEPTS": [
    { title: "SACAgent", href: "/docs/sacagent" },
    { title: "Conformal Prediction", href: "/docs/conformal-prediction" },
    { title: "Offline Learning", href: "/docs/offline-learning" },
  ],
  "API REFERENCE": [
    { title: "SACAgent Class", href: "/docs/api/sacagent" },
    { title: "Configuration", href: "/docs/api/configuration" },
    { title: "Utilities", href: "/docs/api/utilities" },
  ],
}

export default function DocsPage() {
  const [searchQuery, setSearchQuery] = useState("")

  return (
    <div className="relative min-h-screen bg-[#1a1a1a] text-white">
      <StarryBackground />

      {/* Left Sidebar */}
      <div className="fixed top-0 left-0 h-screen w-64 bg-[#0a0a0a]/90 border-r border-white/10 overflow-y-auto">
        <div className="p-4">
          <Link href="/">
            <Button variant="ghost" className="mb-4 text-white/70 hover:text-white w-full justify-start">
              <ChevronLeft className="mr-2 h-4 w-4" />
              Back to Home
            </Button>
          </Link>

          <div className="relative mb-6">
            <Search className="absolute left-2 top-2.5 h-4 w-4 text-white/40" />
            <Input
              type="search"
              placeholder="Search documentation..."
              className="pl-8 bg-white/5 border-white/10 text-white/70 placeholder:text-white/40"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>

          <nav className="space-y-8">
            {Object.entries(sections).map(([category, items]) => (
              <div key={category}>
                <h2 className="text-xs font-semibold text-white/40 mb-2">{category}</h2>
                <ul className="space-y-1">
                  {items.map((item) => (
                    <li key={item.title}>
                      <Link
                        href={item.href}
                        className="block px-2 py-1.5 text-sm text-white/70 hover:text-white hover:bg-white/5 rounded-md"
                      >
                        {item.title}
                      </Link>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </nav>
        </div>
      </div>

      {/* Main Content */}
      <div className="ml-64 mr-64 p-12">
        <div className="max-w-4xl">
          <div className="flex justify-between items-start mb-8">
            <div>
              <h1 className="text-4xl font-bold mb-4">SACAgent Class</h1>
              <p className="text-lg text-white/70">The main interface for working with the Conformal SAC agent.</p>
            </div>
            <Button variant="ghost" className="text-white/70 hover:text-white">
              <Copy className="mr-2 h-4 w-4" />
              Copy page
            </Button>
          </div>

          <div className="prose prose-invert max-w-none">
            <div className="rounded-lg bg-white/5 border border-white/10 p-4 mb-8">
              <p className="text-sm text-white/70">
                The SACAgent class provides a high-level wrapper for training and evaluating agents using the Conformal
                Soft Actor-Critic algorithm with offline datasets.
              </p>
            </div>

            <h2 className="text-2xl font-bold mt-12 mb-4">Initialization</h2>
            <p className="mb-4">Create a new SACAgent instance with the following parameters:</p>

            <div className="bg-[#0a0a0a] rounded-lg p-4 mb-6">
              <pre className="text-sm text-white/90">
                <code>{`from conformal_sac.agent_wrapper import SACAgent

agent = SACAgent(
    env_name="halfcheetah-medium-expert",
    offline=True,
    iteration=100000,
    seed=42,
    learning_rate=3e-4,
    gamma=0.99,
    tau=0.005,
    batch_size=256,
    log_interval=2000,
    alpha_q=100,
    q_alpha_update_freq=50
)`}</code>
              </pre>
            </div>

            <h3 className="text-xl font-semibold mt-8 mb-4">Parameters</h3>
            <ul className="space-y-4">
              <li>
                <code className="bg-white/10 px-2 py-1 rounded text-sm">env_name</code>
                <span className="text-white/70 ml-2">string</span>
                <p className="mt-1 text-white/70">Name of the Gym (or D4RL) environment.</p>
              </li>
              <li>
                <code className="bg-white/10 px-2 py-1 rounded text-sm">offline</code>
                <span className="text-white/70 ml-2">boolean, default: True</span>
                <p className="mt-1 text-white/70">If True, use an offline dataset from D4RL.</p>
              </li>
              <li>
                <code className="bg-white/10 px-2 py-1 rounded text-sm">iteration</code>
                <span className="text-white/70 ml-2">integer, default: 100000</span>
                <p className="mt-1 text-white/70">Number of training iterations.</p>
              </li>
            </ul>

            <h2 className="text-2xl font-bold mt-12 mb-4">Methods</h2>

            <h3 className="text-xl font-semibold mt-8 mb-4">train()</h3>
            <p className="mb-4 text-white/70">
              Runs the training loop. During training, the agent's update method is called repeatedly. Evaluation is
              performed every log_interval steps.
            </p>
            <div className="bg-[#0a0a0a] rounded-lg p-4 mb-6">
              <pre className="text-sm text-white/90">
                <code>{`# Train the agent
agent.train()`}</code>
              </pre>
            </div>

            <h3 className="text-xl font-semibold mt-8 mb-4">evaluate(eval_episodes: int = 5) → float</h3>
            <p className="mb-4 text-white/70">Evaluates the current policy on the environment.</p>
            <div className="bg-[#0a0a0a] rounded-lg p-4 mb-6">
              <pre className="text-sm text-white/90">
                <code>{`# Evaluate the agent
score = agent.evaluate(eval_episodes=5)
print(f"Final evaluation score: {score}")`}</code>
              </pre>
            </div>
          </div>
        </div>
      </div>

      {/* Right Sidebar */}
      <div className="fixed top-0 right-0 h-screen w-64 bg-[#0a0a0a]/90 border-l border-white/10 p-4">
        <h2 className="text-sm font-semibold text-white/40 mb-4">ON THIS PAGE</h2>
        <ul className="space-y-2 text-sm">
          <li>
            <a href="#initialization" className="text-white/70 hover:text-white">
              Initialization
            </a>
          </li>
          <li>
            <a href="#parameters" className="text-white/70 hover:text-white ml-4">
              Parameters
            </a>
          </li>
          <li>
            <a href="#methods" className="text-white/70 hover:text-white">
              Methods
            </a>
          </li>
          <li>
            <a href="#train" className="text-white/70 hover:text-white ml-4">
              train()
            </a>
          </li>
          <li>
            <a href="#evaluate" className="text-white/70 hover:text-white ml-4">
              evaluate()
            </a>
          </li>
        </ul>
      </div>
    </div>
  )
}

