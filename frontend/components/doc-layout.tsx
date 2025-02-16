"use client"

import React, { useState } from "react"
import Link from "next/link"
import { Search, ChevronLeft } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"

const sections = {
  "GET STARTED": [
    { title: "Overview", href: "/docs/overview" },
    { title: "Quickstart", href: "/docs/quickstart" },
  ],
  "CORE CONCEPTS": [
    { title: "Conformal Q-Learning", href: "/docs/conformal-q-learning" },
    { title: "Conformal Prediction", href: "/docs/conformal-prediction" },
    { title: "Offline Learning", href: "/docs/offline-learning" },
  ],
  "API REFERENCE": [
    { title: "ConformalQLearning", href: "/docs/api/conformal-q-learning" },
    { title: "Configuration", href: "/docs/api/configuration" },
    { title: "Utilities", href: "/docs/api/utilities" },
  ],
}

function StarryBackground() {
  const [stars, setStars] = useState<Array<{ id: number; x: number; y: number; size: number; delay: number }>>([])

  React.useEffect(() => {
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
          <circle
            key={star.id}
            cx={star.x}
            cy={star.y}
            r={star.size}
            fill="white"
            opacity={0.1 + Math.random() * 0.5}
          />
        ))}
      </svg>
    </div>
  )
}

export default function DocLayout({ children }: { children: React.ReactNode }) {
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
      <div className="ml-64 p-12">
        <div className="max-w-4xl mx-auto">{children}</div>
      </div>
    </div>
  )
}

