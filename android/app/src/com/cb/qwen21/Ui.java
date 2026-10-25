package com.cb.qwen21;

import android.content.Context;
import android.graphics.Color;
import android.graphics.drawable.GradientDrawable;
import android.text.InputType;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;

final class Ui {
    static final int BG = 0xFF0E1116, CARD = 0xFF171C24, ACCENT = 0xFF4EC9B0,
            TEXT = 0xFFE8EDF3, MUTED = 0xFF8A94A6, LINE = 0xFF232A35;

    private Ui() {}

    static int dp(Context c, float v) {
        return (int) TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v, c.getResources().getDisplayMetrics());
    }

    static GradientDrawable rounded(Context c, int color, float r, int stroke) {
        GradientDrawable d = new GradientDrawable();
        d.setColor(color);
        d.setCornerRadius(dp(c, r));
        if (stroke != 0) d.setStroke(dp(c, 1), stroke);
        return d;
    }

    static TextView label(Context c, String s, int size, int color) {
        TextView t = new TextView(c);
        t.setText(s);
        t.setTextColor(color);
        t.setTextSize(size);
        return t;
    }

    static TextView card(Context c, String title) {
        TextView t = label(c, title, 15, TEXT);
        t.setTypeface(null, android.graphics.Typeface.BOLD);
        return t;
    }

    static EditText input(Context c, String hint, String value, boolean multiline) {
        EditText e = new EditText(c);
        e.setHint(hint);
        e.setText(value);
        e.setTextColor(TEXT);
        e.setHintTextColor(MUTED);
        e.setTextSize(14);
        e.setBackground(rounded(c, 0xFF10151C, 10, LINE));
        e.setPadding(dp(c, 12), dp(c, 10), dp(c, 12), dp(c, 10));
        e.setInputType(multiline
                ? InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_MULTI_LINE
                : InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_URI);
        if (multiline) {
            e.setMinLines(3);
            e.setGravity(Gravity.TOP | Gravity.START);
        } else {
            e.setSingleLine(true);
        }
        return e;
    }

    static LinearLayout column(Context c) {
        LinearLayout l = new LinearLayout(c);
        l.setOrientation(LinearLayout.VERTICAL);
        return l;
    }

    static LinearLayout panel(Context c) {
        LinearLayout l = column(c);
        l.setBackground(rounded(c, CARD, 16, LINE));
        int p = dp(c, 14);
        l.setPadding(p, p, p, p);
        return l;
    }

    static void gap(View v, int px) {
        ViewGroup.LayoutParams lp = v.getLayoutParams();
        if (lp instanceof LinearLayout.LayoutParams) ((LinearLayout.LayoutParams) lp).topMargin = px;
    }

    static void add(LinearLayout parent, View child, int topMarginDp) {
        if (topMarginDp > 0) {
            LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
            lp.topMargin = dp(parent.getContext(), topMarginDp);
            parent.addView(child, lp);
        } else {
            parent.addView(child, new LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        }
    }
}
