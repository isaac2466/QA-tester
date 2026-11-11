import { useEffect, useId, useRef, useState } from "react";
import { createPortal } from "react-dom";

export default function Dropdown({ label, value, options, onChange, className = "" }) {
  const id = useId();
  const root = useRef(null);
  const trigger = useRef(null);
  const menu = useRef(null);
  const optionRefs = useRef([]);
  const selectedIndex = Math.max(0, options.findIndex((option) => option.value === value));
  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(selectedIndex);
  const [menuPosition, setMenuPosition] = useState(null);
  const selected = options.find((option) => option.value === value);

  useEffect(() => {
    if (!open) return;
    optionRefs.current[activeIndex]?.focus();
  }, [open, activeIndex]);

  useEffect(() => {
    if (!open) return;
    const updatePosition = () => {
      const rect = trigger.current.getBoundingClientRect();
      const height = Math.min(options.length * 34 + 8, 240);
      const spaceBelow = window.innerHeight - rect.bottom - 8;
      const openAbove = spaceBelow < height && rect.top > height;
      const top = openAbove ? rect.top - height - 4 : rect.bottom + 4;
      setMenuPosition({
        top,
        left: rect.left,
        minWidth: rect.width,
        maxHeight: openAbove ? Math.min(240, rect.top - 16) : Math.max(100, spaceBelow),
      });
    };
    updatePosition();
    window.addEventListener("resize", updatePosition);
    window.addEventListener("scroll", updatePosition, true);
    return () => {
      window.removeEventListener("resize", updatePosition);
      window.removeEventListener("scroll", updatePosition, true);
    };
  }, [open, options.length]);

  useEffect(() => {
    if (!open) return;
    const closeOutside = (event) => {
      if (!root.current?.contains(event.target) && !menu.current?.contains(event.target)) setOpen(false);
    };
    document.addEventListener("pointerdown", closeOutside);
    return () => document.removeEventListener("pointerdown", closeOutside);
  }, [open]);

  const openAt = (index = selectedIndex) => {
    setActiveIndex(index);
    setOpen(true);
  };
  const close = () => {
    setOpen(false);
    trigger.current?.focus();
  };
  const move = (index) => {
    const next = (index + options.length) % options.length;
    setActiveIndex(next);
    optionRefs.current[next]?.focus();
  };
  const onTriggerKeyDown = (event) => {
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      openAt(event.key === "ArrowDown" ? selectedIndex : (selectedIndex + options.length - 1) % options.length);
    } else if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      if (open) close();
      else openAt();
    }
  };
  const onOptionKeyDown = (event, index) => {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      move(index + 1);
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      move(index - 1);
    } else if (event.key === "Home") {
      event.preventDefault();
      move(0);
    } else if (event.key === "End") {
      event.preventDefault();
      move(options.length - 1);
    } else if (event.key === "Escape") {
      event.preventDefault();
      close();
    } else if (event.key === "Tab") {
      setOpen(false);
    } else if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onChange(options[index].value);
      close();
    }
  };

  return (
    <div className={`custom-select ${className}`} ref={root}>
      <button
        ref={trigger}
        type="button"
        className="select-trigger"
        aria-label={label}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={id}
        onClick={() => open ? close() : openAt()}
        onKeyDown={onTriggerKeyDown}
      >
        {selected?.label}
        <span className={`select-chevron ${open ? "open" : ""}`} aria-hidden="true">▾</span>
      </button>
      {open && menuPosition && createPortal(
        <div ref={menu} className="select-menu" id={id} role="listbox" aria-label={label} style={menuPosition}>
          {options.map((option, index) => (
            <div
              key={option.value}
              ref={(element) => { optionRefs.current[index] = element; }}
              className="select-option"
              role="option"
              aria-selected={option.value === value}
              tabIndex={-1}
              onMouseMove={() => setActiveIndex(index)}
              onClick={() => { onChange(option.value); close(); }}
              onKeyDown={(event) => onOptionKeyDown(event, index)}
            >
              {option.label}
            </div>
          ))}
        </div>,
        document.body,
      )}
    </div>
  );
}
