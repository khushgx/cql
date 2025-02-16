"use client"

import { useState, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Button } from "@/components/ui/button"
import { Github } from "lucide-react"
import Link from "next/link"

function StarryBackground() {
  const [hoveredStar, setHoveredStar] = useState<number | null>(null)
  const [stars, setStars] = useState<Array<{ id: number; x: number; y: number; size: number; delay: number }>>([])

  useEffect(() => {
    // Create 200 stars with random positions, sizes, and animation delays
    const newStars = Array.from({ length: 200 }, (_, i) => ({
      id: i,
      x: Math.random() * 100,
      y: Math.random() * 100,
      size: Math.random() * 0.15 + 0.05, // Sizes between 0.05 and 0.2
      delay: Math.random() * 3, // Random delay for twinkling effect
    }))
    setStars(newStars)
  }, [])

  return (
    <div className="absolute inset-0 pointer-events-none">
      <svg className="w-full h-full" viewBox="0 0 100 100">
        <defs>
          <radialGradient id="starGlow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="white" stopOpacity="1" />
            <stop offset="100%" stopColor="white" stopOpacity="0" />
          </radialGradient>
        </defs>
        {stars.map((star) => (
          <g key={star.id}>
            <motion.circle
              cx={star.x}
              cy={star.y}
              r={star.size}
              fill="white"
              initial={{ opacity: 0.1 }}
              animate={{
                opacity: [0.2, 0.8, 0.2],
                scale: hoveredStar === star.id ? [1, 1.5, 1] : [1, 1.2, 1],
              }}
              transition={{
                duration: 2 + star.delay,
                repeat: Number.POSITIVE_INFINITY,
                ease: "easeInOut",
              }}
              onMouseEnter={() => setHoveredStar(star.id)}
              onMouseLeave={() => setHoveredStar(null)}
            />
            <AnimatePresence>
              {hoveredStar === star.id && (
                <motion.circle
                  cx={star.x}
                  cy={star.y}
                  r={star.size * 4}
                  fill="url(#starGlow)"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 0.5 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.3 }}
                />
              )}
            </AnimatePresence>
          </g>
        ))}
      </svg>
    </div>
  )
}

export default function LandingPage() {
  const title = "RL-CP Fusion"
  const words = title.split(" ")

  return (
    <div className="relative min-h-screen w-full flex items-center justify-center overflow-hidden bg-[#0a0a0a]">
      <StarryBackground />

      <div className="relative z-10 container mx-auto px-4 md:px-6 text-center">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 2 }}
          className="max-w-4xl mx-auto"
        >
          <h1 className="text-5xl sm:text-7xl md:text-8xl font-bold mb-8 tracking-tighter">
            {words.map((word, wordIndex) => (
              <span key={wordIndex} className="inline-block mr-4 last:mr-0">
                {word.split("").map((letter, letterIndex) => (
                  <motion.span
                    key={`${wordIndex}-${letterIndex}`}
                    initial={{ y: 100, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{
                      delay: wordIndex * 0.1 + letterIndex * 0.03,
                      type: "spring",
                      stiffness: 150,
                      damping: 25,
                    }}
                    className="inline-block text-transparent bg-clip-text bg-gradient-to-r from-white to-gray-300"
                  >
                    {letter}
                  </motion.span>
                ))}
              </span>
            ))}
          </h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.5 }}
            className="text-xl sm:text-2xl text-gray-300 max-w-2xl mx-auto mb-8"
          >
            Offline Reinforcement Learning with Conformal Prediction
          </motion.p>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.7 }}
            className="text-lg text-gray-400 max-w-3xl mx-auto mb-12"
          >
            Tackle real-world decision-making challenges with our innovative approach that combines the power of Offline
            Reinforcement Learning and the reliability of Conformal Prediction.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.9 }}
            className="flex flex-col sm:flex-row gap-4 justify-center mb-16"
          >
            <div className="group relative">
              <div className="absolute -inset-0.5 bg-gradient-to-r from-white to-gray-400 rounded-2xl blur opacity-30 group-hover:opacity-60 transition duration-500"></div>
              <Button
                variant="ghost"
                className="relative rounded-[1.15rem] px-8 py-6 text-lg font-semibold bg-black/90 text-white 
                          transition-all duration-300 group-hover:-translate-y-1 border border-white/10"
                onClick={() => window.open("https://github.com/khushgx/cql", "_blank")}
              >
                <Github className="mr-2 h-5 w-5" />
                <span className="opacity-90 group-hover:opacity-100 transition-opacity">View on GitHub</span>
                <span className="ml-3 opacity-70 group-hover:opacity-100 group-hover:translate-x-1.5 transition-all duration-300">
                  →
                </span>
              </Button>
            </div>
            <div className="group relative">
              <div className="absolute -inset-0.5 bg-gradient-to-r from-white to-gray-400 rounded-2xl blur opacity-30 group-hover:opacity-60 transition duration-500"></div>
              <Link href="/docs">
                <Button
                  variant="ghost"
                  className="relative rounded-[1.15rem] px-8 py-6 text-lg font-semibold bg-black/90 text-white 
                            transition-all duration-300 group-hover:-translate-y-1 border border-white/10"
                >
                  <span className="opacity-90 group-hover:opacity-100 transition-opacity">Read the Docs</span>
                  <span className="ml-3 opacity-70 group-hover:opacity-100 group-hover:translate-x-1.5 transition-all duration-300">
                    →
                  </span>
                </Button>
              </Link>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1, delay: 1.1 }}
            className="grid grid-cols-1 md:grid-cols-3 gap-8"
          >
            {[
              {
                title: "Offline Learning",
                description: "Learn effective policies from static datasets without direct environment interaction.",
              },
              {
                title: "Uncertainty Quantification",
                description: "Leverage Conformal Prediction for robust confidence intervals on Q-value estimates.",
              },
              {
                title: "Stable & Reliable",
                description: "Mitigate overestimation risks and ensure a more stable learning process.",
              },
            ].map((feature, index) => (
              <div key={index} className="group relative">
                <div className="absolute -inset-0.5 bg-gradient-to-r from-white to-gray-400 rounded-2xl blur opacity-20 group-hover:opacity-40 transition duration-500"></div>
                <div
                  className="relative p-6 rounded-[1.15rem] bg-black/90 text-white transition-all duration-300 
                               group-hover:-translate-y-1 border border-white/10 h-full"
                >
                  <h3 className="text-xl font-semibold mb-2 text-gray-100">{feature.title}</h3>
                  <p className="text-gray-300">{feature.description}</p>
                </div>
              </div>
            ))}
          </motion.div>
        </motion.div>
      </div>
    </div>
  )
}

